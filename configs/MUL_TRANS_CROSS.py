from configs import BaseConfig


class MUL_TRANS_CROSS_Config(BaseConfig):
    def __init__(self, **kwargs) -> None:
        remaining_args = super().__init__(**kwargs)
        self.set_default_config()
        self.update(remaining_args)

    def set_default_config(self):
        # common configs
        self.early_stop = 6
        self.finetune_bert = True
        self.coupled_instance = False   # Whether paired (clean, noisy) instances are provided in the training.
        # dataset specific configs
        if self.dataset == "MOSI":
            self.pretrained_bert_model = "/home/zhangbaozheng/paper_code/Robust_Framework/pretrained_model/bert_en"
            self.feature_dims = [768, 25, 171]
            self.hidden_dims = 256
            self.fus_nheads = 8
            self.fus_layers = 1
            self.fus_attn_mask = False
            self.fus_position_embedding = True
            self.fus_relu_dropout = 0.0
            self.fus_embed_dropout = 0.5
            self.fus_res_dropout = 0.4
            self.fus_attn_dropout = 0.5
            self.post_fusion_dropout = 0.2
            self.augmentation = ["rawa_color_w","rawv_gblur","rawv_impulse_value","rawa_bg_park"]
            self.batch_size = 32
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-05
            self.weight_decay_bert = 0.0005
            self.learning_rate_other = 0.0001
            self.weight_decay_other = 0.0005
            self.grad_clip = 0.6
            
        elif self.dataset == 'SIMSv2': 
            # NOTE  just a copy and paste of SIMSv2 Hyperparameters w.o. tuning.
            self.pretrained_bert_model = '/home/zhangbaozheng/paper_code/Robust_Framework/pretrained_model/bert_cn'
            self.feature_dims = [768, 25, 25]
            self.hidden_dims = 256
            self.fus_nheads = 8
            self.trans_nheads = 8
            self.fus_layers = 1
            self.trans_layers = 5
            self.fus_attn_mask = False
            self.fus_position_embedding = True
            self.fus_relu_dropout = 0.0
            self.fus_embed_dropout = 0.5
            self.fus_res_dropout = 0.4
            self.fus_attn_dropout = 0.5
            self.post_fusion_dropout = 0.2
            self.augmentation = ["feat_random_drop"]
            self.batch_size = 128
            
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-05
            self.weight_decay_bert = 0.0005
            self.learning_rate_other = 0.0001
            self.learning_rate_recon = 0.0001
            self.weight_decay_other = 0.0005
            self.grad_clip = 0.6
        else:
            self.logger.warning(f"No default config for dataset {self.dataset}")