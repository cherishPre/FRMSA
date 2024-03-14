import torch
from tqdm import tqdm
from torch import optim
from utils.functions import dict_to_str
from trainers.base_trainer import BaseTrainer

class MUL_TRANS_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def do_train(self, model, dataloader):
        if self.config.finetune_bert:
            # OPTIMIZER: finetune Bert Parameters.
            bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            bert_params = list(model.text_model.named_parameters())
            bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
            bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
            model_params_other = [p for n, p in list(model.named_parameters()) if not (('text_model' in n) or ('recon_feature' in n))]
            recon_params = [p for n, p in list(model.named_parameters()) if 'recon_feature' in n]

            optimizer_grouped_parameters = [
                {'params': bert_params_decay, 'weight_decay': self.config.weight_decay_bert, 'lr': self.config.learning_rate_bert},
                {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.config.learning_rate_bert},
                {'params': model_params_other, 'weight_decay': self.config.weight_decay_other, 'lr': self.config.learning_rate_other}
            ]
            optimizer = optim.Adam(optimizer_grouped_parameters)

            recon_optimizer = optim.Adam([
                {'params': recon_params, 'weight_decay': self.config.weight_decay_other, 'lr': self.config.learning_rate_recon}
            ])
        else:
            model_params_other = [p for n, p in list(model.named_parameters()) if not (('text_model' in n) or ('recon_feature' in n))]
            recon_params = [p for n, p in list(model.named_parameters()) if 'recon_feature' in n]

            optimizer_grouped_parameters = [
                {'params': model_params_other, 'weight_decay': self.config.weight_decay_other, 'lr': self.config.learning_rate_other}
            ]
            optimizer = optim.Adam(optimizer_grouped_parameters)

            recon_optimizer = optim.Adam([
                {'params': recon_params, 'weight_decay': self.config.weight_decay_other, 'lr': self.config.learning_rate_recon}
            ])
        # loop util earlystop
        
        while True: 
            self.epochs += 1
            # train
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    text = batch_data['text'].to(self.config.device)
                    text_m = batch_data['text_m'].to(self.config.device) # Paired Noisy data
                    audio = batch_data['audio'].to(self.config.device)
                    audio_m = batch_data['audio_m'].to(self.config.device) # Paired Noisy data
                    vision = batch_data['vision'].to(self.config.device)
                    vision_m = batch_data['vision_m'].to(self.config.device) # Paired Noisy data
                    vision_lengths = batch_data['vision_lengths'].to(self.config.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.config.device)

                    labels = batch_data['labels']['M'].to(self.config.device)
                    labels = labels.view(-1, 1)

                    segments = torch.tensor([[[0] * self.config.hidden_dims] * self.config.seq_lens[0] + [[1] * 
                            self.config.hidden_dims] * self.config.seq_lens[0] + [[2] * self.config.hidden_dims] * self.config.seq_lens[0]] * 
                        text.shape[0], requires_grad=False).to(self.config.device)
                    # clear gradient

                    optimizer.zero_grad()
                    # compute loss, backward, update.
                    result = model.phase_one(
                        text,
                        audio,
                        vision,
                        audio_lengths, 
                        vision_lengths,
                        segments
                    )
                    prediction = result['prediction']
                    loss = self.criterion(prediction, labels)
                    loss.backward()
                    optimizer.step()

                    optimizer.zero_grad()
                    recon_optimizer.zero_grad()
                    result = model.phase_two(
                        (text, text_m),
                        (audio, audio_m),
                        (vision, vision_m),
                        audio_lengths,
                        vision_lengths,
                        segments
                    )
                    prediction = result['prediction']
                    loss = self.criterion(prediction, labels)
                    loss += result['rloss']
                    loss.backward()
                    recon_optimizer.step()
                    optimizer.step()
                    
                    # optimizer.zero_grad()
                    # recon_optimizer.zero_grad()
                    # result = model.phase_three(
                    #     text_m,
                    #     audio_m,
                    #     vision_m,
                    #     audio_lengths, 
                    #     vision_lengths,
                    #     segments
                    # )

                    # prediction = result['prediction']
                    # loss = self.criterion(prediction, labels)
                    # loss.backward()
                    # optimizer.step()
                    # recon_optimizer.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())
                    
            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            self.logger.info(
                f"TRAIN-({self.config.model}) [{self.epochs - self.best_epoch}/{self.epochs}/{self.config.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_valid(model, dataloader['valid'])
            # save best model
            cur_valid = val_results[self.config.KeyEval]
            isBetter = cur_valid <= (self.best_valid - 1e-6)
            # save best model
            if isBetter:
                self.best_valid, self.best_epoch = cur_valid, self.epochs
                # save model
                torch.save(model.cpu().state_dict(), self.config.model_save_path)
                model.to(self.config.device)
            if self.epochs - self.best_epoch >= self.config.early_stop:
                return None


    def do_valid(self, model, dataloader, mode='EVAL'):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    text = batch_data['text'].to(self.config.device)
                    audio = batch_data['audio'].to(self.config.device)
                    vision = batch_data['vision'].to(self.config.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.config.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.config.device)
                    labels = batch_data['labels']['M'].to(self.config.device)
                    labels = labels.view(-1, 1)
                    result = model(
                        text, 
                        audio, 
                        vision, 
                        audio_lengths, 
                        vision_lengths
                    )

                    prediction = result['prediction']
                    prediction = torch.where(
                    torch.isnan(prediction),
                    torch.full_like(prediction, 0),
                    prediction)
                    loss = self.criterion(prediction, labels)
                    eval_loss += loss.item()
                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        self.logger.info(f"{mode}-({self.config.model}) >> {dict_to_str(eval_results)}")

        return eval_results
    
