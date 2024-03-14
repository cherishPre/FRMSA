import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import BaseConfig
from models.subnets.BertTextEncoder import BertTextEncoder
from models.subnets.AlignSubNet import AlignSubNet
from models.subnets.transformers_encoder.transformer import TransformerEncoder

class FeatureReconstructionLoss(nn.Module):
    def __init__(self,hidden_dims):
        super(FeatureReconstructionLoss, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=[150,hidden_dims], elementwise_affine=False)

    def forward(self, input_features, reconstructed_features):
        loss = torch.mean(torch.absolute(self.layernorm(input_features) - self.layernorm(reconstructed_features)))
        return loss
    
class CNNFusionBlock(nn.Module):
    def __init__(self, input_dim, in_channels=1, out_channels=256, kernel_heights=[3,4,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        embd_size = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels, embd_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        return embd
 
# The fusion network and the classification network
class MUL_TRANS(nn.Module):
    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config

        self.reconLoss = FeatureReconstructionLoss(config.hidden_dims)
        # BERT SUBNET FOR TEXT
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert)

        # Alignment Network
        self.alignment_network = AlignSubNet(config, mode='ctc')

        # preprocess network
        self.text_unity_dim = nn.Linear(config.feature_dims[0], config.hidden_dims)
        self.audio_unity_dim = nn.Linear(config.feature_dims[1], config.hidden_dims)
        self.video_unity_dim = nn.Linear(config.feature_dims[2], config.hidden_dims)

        self.recon_feature = TransformerEncoder(embed_dim=config.hidden_dims, num_heads=config.trans_nheads, layers=config.trans_layers,
                                                attn_dropout=config.fus_attn_dropout, relu_dropout=config.fus_relu_dropout, res_dropout=config.fus_res_dropout,
                                                embed_dropout=config.fus_embed_dropout, attn_mask=config.fus_attn_mask, position_embedding = config.fus_position_embedding)

        self.fusion_trans = TransformerEncoder(embed_dim=config.hidden_dims, num_heads=config.fus_nheads, layers=config.fus_layers,
                                                attn_dropout=config.fus_attn_dropout, relu_dropout=config.fus_relu_dropout, res_dropout=config.fus_res_dropout,
                                                embed_dropout=config.fus_embed_dropout, attn_mask=config.fus_attn_mask, position_embedding = config.fus_position_embedding)
        
        self.fusion = CNNFusionBlock(config.hidden_dims,out_channels=config.hidden_dims)

        self.post_fusion_dropout = nn.Dropout(p=config.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(config.hidden_dims, int(config.hidden_dims/2))
        self.post_fusion_layer_2 = nn.Linear(int(config.hidden_dims/2), 1)

    
    def phase_one(self, text, audio, vision, audio_lengths, video_lengths, segments):
        text = self.text_model(text)
        text, audio, video = self.alignment_network(text, audio, vision, 
                                                        audio_lengths, video_lengths)
        text = self.text_unity_dim(text)
        audio = self.audio_unity_dim(audio)
        video = self.video_unity_dim(video)
        fusion = torch.cat((text, audio, video), dim=1)
        fusion = self.fusion_trans(fusion,segments)

        fusion = self.fusion(fusion)

        fusion_post = self.post_fusion_dropout(fusion)
        fusion_post = F.relu(self.post_fusion_layer_1(fusion_post), inplace=False)
        prediction = self.post_fusion_layer_2(fusion_post)
        result ={
            'prediction': prediction
        }
        return result

    def phase_two(self, text, audio, vision, audio_lengths, video_lengths, segments):
        text, text_m = text
        audio, audio_m = audio 
        vision, vision_m= vision

        text_m = self.text_model(text_m)
        text_m, audio_m, video_m = self.alignment_network(text_m, audio_m, vision_m, 
                                                        audio_lengths, video_lengths)
        text_m = self.text_unity_dim(text_m)
        audio_m = self.audio_unity_dim(audio_m)
        video_m = self.video_unity_dim(video_m)
        fusion_m = torch.cat((text_m, audio_m, video_m), dim=1)
        
        with torch.no_grad():
            text = self.text_model(text)
            text, audio, video = self.alignment_network(text, audio, vision, 
                                                            audio_lengths, video_lengths)
            text = self.text_unity_dim(text)
            audio = self.audio_unity_dim(audio)
            video = self.video_unity_dim(video)
            fusion = torch.cat((text, audio, video), dim=1)

        fusion_m = self.recon_feature(fusion_m,segments)
        rloss = self.reconLoss(fusion_m,fusion)

        fusion_m = self.fusion_trans(fusion_m, segments)
        fusion_m = self.fusion(fusion_m)

        fusion_post = self.post_fusion_dropout(fusion_m)
        fusion_post = F.relu(self.post_fusion_layer_1(fusion_post), inplace=False)
        prediction = self.post_fusion_layer_2(fusion_post)
        result ={
            'prediction': prediction,
            'rloss':rloss
        }
        return result
    
    def forward(self, text, audio, vision, audio_lengths, video_lengths):
        '''
        Args:
            audio: tensor of shape (batch_size, sequence_len, audio_in)
            vision: tensor of shape (batch_size, sequence_len, video_in)
            text: tensor of shape (batch_size, sequence_len, text_in)
        '''
        segments = torch.tensor([[[0] * self.config.hidden_dims] * self.config.seq_lens[0] + [[1] * 
                            self.config.hidden_dims] * self.config.seq_lens[0] + [[2] * self.config.hidden_dims] * self.config.seq_lens[0]] * 
                        text.shape[0], requires_grad=False).to(self.config.device)
        
        text = self.text_model(text)
        text, audio, video = self.alignment_network(text, audio, vision, 
                                                        audio_lengths, video_lengths)
        text = self.text_unity_dim(text)
        audio = self.audio_unity_dim(audio)
        video = self.video_unity_dim(video)
        fusion = torch.cat((text, audio, video), dim=1)
        fusion = self.recon_feature(fusion,segments)

        fusion = self.fusion_trans(fusion, segments)
        fusion = self.fusion(fusion)

        fusion_post = self.post_fusion_dropout(fusion)
        fusion_post = F.relu(self.post_fusion_layer_1(fusion_post), inplace=False)
        prediction = self.post_fusion_layer_2(fusion_post)
        result ={
            'prediction': prediction
        }
        return result
