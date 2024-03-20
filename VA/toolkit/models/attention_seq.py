'''
Description: unimodal encoder + concat + attention fusion
'''
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder, LSTM_Seq


class AttentionSeq(nn.Module):
    def __init__(self, args):
        super(AttentionSeq, self).__init__()

        text_dim = args.text_dim
        audio_dim = args.audio_dim
        video_dim = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        seq_len= args.seq_len
        self.grad_clip = args.grad_clip

        self.audio_encoder = LSTM_Seq(audio_dim, hidden_dim, dropout)
        self.text_encoder = LSTM_Seq(text_dim, hidden_dim, dropout)
        self.video_encoder = LSTM_Seq(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3 , hidden_dim, dropout)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.fc_att = nn.Linear(hidden_dim, 3)
        # self.fc_out_1 = nn.Linear(hidden_dim * 2, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim * 2, output_dim2)

    def forward(self, batch):
        '''
            support feat_type: utt | frm-align | frm-unalign
        '''
        # print('input_size:',batch['audios'].size(), batch['texts'].size(), batch['videos'].size())
        self.lstm.flatten_parameters()
        audio_hidden = self.audio_encoder(batch['audios'])  # [32,200,128]
        text_hidden = self.text_encoder(batch['texts'])  # [32,200,128]
        video_hidden = self.video_encoder(batch['videos'])  # [32,200,128]
        # print('encoder_size:',audio_hidden.size(), text_hidden.size(), video_hidden.size())

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=2)  # [32,200,384]
        # print('multi_hidden1_size:',multi_hidden1.size())
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 3)  # [32, 200, 3, 1]
        # print('attention:', attention.size())

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=3)  # [32,200,128, 3]
        # print('multi_hidden2:',multi_hidden2.size())
        fused_feat = torch.matmul(multi_hidden2, attention)  # [32, 200, 128, 3] * [32, 200, 3, 1] = [32, 200, 128, 1]
        # print('fused_feat:',fused_feat.size())

        fused_feat = fused_feat.squeeze(axis=3)  # [32,200,128] => 解决batch=1报错的问题
        # print('fused_feat:',fused_feat.size())
        features,_= self.lstm(fused_feat)#[32,200,256] 双向lstm，维度翻倍
        # print('features:',features.size())
        # emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        # print('val_out:',vals_out.size())
        interloss = torch.tensor(0).cuda()

        # return features, emos_out, vals_out, interloss
        return features, _, vals_out, interloss
