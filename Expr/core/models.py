"""
Author: Van-Thong Huynh
Department of AI Convergence, Chonnam Natl. Univ.
"""
import models_vit
#models_vit.path.append('./tools/models_vit.py')
import timm
import torch.nn
from pytorch_lightning import LightningModule
from torchmetrics import F1Score, PearsonCorrCoef, MeanSquaredError
from pl_bolts.optimizers import lr_scheduler as pl_lr_scheduler
from torch.optim import lr_scheduler
from core.metrics import ConCorrCoef

import torch

from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.models import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
from torch.nn import functional as F

from core.config import cfg
from core.loss import CCCLoss, CELogitLoss, BCEwithLogitsLoss, MSELoss, SigmoidFocalLoss, CEFocalLoss

from pretrained import vggface2
from pretrained.facex_zoo import get_facex_zoo
from functools import partial
import math
from transformers import ViTModel

from einops.layers.torch import Rearrange
import torch
from core.model import Attention, Transformer, FeedForward, LightweightModel
# Facenet https://github.com/timesler/facenet-pytorch
########################################################################################################################################
torch.use_deterministic_algorithms(True)

def get_vggface2(model_name):
    if 'senet50' in model_name:
        vgg2_model = vggface2.resnet50(include_top=False, se=True, num_classes=8631)
        vgg2_ckpt = torch.load('pretrained/vggface2_weights/senet50_ft_weight.pth')
    elif 'resnet50' in model_name:
        vgg2_model = vggface2.resnet50(include_top=False, se=False, num_classes=8631)
        vgg2_ckpt = torch.load('pretrained/vggface2_weights/resnet50_ft_weight.pth')
    else:
        raise ValueError('Unkown model name {} for VGGFACE2'.format(model_name))

    vgg2_model.load_state_dict(vgg2_ckpt['model_state_dict'])
    return vgg2_model


class ABAW3Model(LightningModule):

    def get_backbone(self):
        """
        https://pytorch.org/vision/master/generated/torchvision.models.feature_extraction.create_feature_extractor.html
        :return:
        """
        # TODO: Custom backbone, freeze layers
        backbone_name = self.backbone_name
        if 'regnet' in backbone_name:
            # REGNET - IMAGENET
            regnet_backbone_dict = {'400mf': (regnet_y_400mf, 440), '800mf': (regnet_y_800mf, 784),
                                    '1.6gf': (regnet_y_1_6gf, 888), '3.2gf': (regnet_y_3_2gf, 1512)}

            bb_model = regnet_backbone_dict[backbone_name.split('-')[-1]][0](pretrained=True)
            backbone = create_feature_extractor(bb_model, return_nodes={'flatten': 'feat'})
            # regnet_y: 400mf = 440, 800mf = 784, regnet_y_1_6gf = 888
            # regnet_x: 400mf = 400
            num_feats = {'feat': regnet_backbone_dict[backbone_name.split('-')[-1]][1]}

            if len(self.backbone_freeze) > 0:
                # Freeze backbone model
                for named, param in backbone.named_parameters():
                    do_freeze = True
                    if 'all' not in cfg.MODEL.BACKBONE_FREEZE or not (
                            isinstance(param, nn.BatchNorm2d) and cfg.MODEL.FREEZE_BATCHNORM):
                        for layer_name in self.backbone_freeze:
                            if layer_name in named:
                                do_freeze = False
                                break
                    if do_freeze:
                        param.requires_grad = False

            return backbone, num_feats

        elif backbone_name in ['vggface2-senet50', 'vggface2-resnet50']:
            # PRETRAINED on VGGFACE2
            bb_model = get_vggface2(cfg.MODEL.BACKBONE)
            backbone = create_feature_extractor(bb_model, return_nodes={'flatten': 'feat'})
            num_feats = {'feat': 2048}

            # Freeze backbone model
            for named, param in backbone.named_parameters():
                do_freeze = True
                if 'all' not in cfg.MODEL.BACKBONE_FREEZE or not (
                        isinstance(param, nn.BatchNorm2d) and cfg.MODEL.FREEZE_BATCHNORM):
                    for layer_name in cfg.MODEL.BACKBONE_FREEZE:
                        if layer_name in named:
                            do_freeze = False
                            break
                if do_freeze:
                    param.requires_grad = False

            return backbone, num_feats
           
        elif backbone_name.split('.')[0] == 'facex':
            # FaceX-Zoo models
            bb_model = get_facex_zoo(backbone_name.split('.')[1],
                                     root_weights='/home/hvthong/sXProject/Affwild2_ABAW3/pretrained/facex_zoo')
            # bn is batch norm of Linear layer
            return_node = 'bn' if 'MobileFaceNet' in cfg.MODEL.BACKBONE else 'output_layer'
            backbone = create_feature_extractor(bb_model, return_nodes={return_node: 'feat'})
            num_feats = {'feat': 512}

            if cfg.MODEL.BACKBONE_FREEZE:
                # Freeze backbone model, layer1, layer2, layer3, layer4
                for named, param in backbone.named_parameters():
                    if 'layer9999999' not in named:
                        param.requires_grad = False

            return backbone, num_feats
        
        elif backbone_name == "mobilenetv3":
            backbone = timm.create_model(
                'timm/mobilenetv3_large_100.ra_in1k',
                pretrained=False,
                num_classes=0,  # remove classifier nn.Linear
            )
            checkpoint_path="/data/zhangfengyu/models/mobilenetv3_large_100.ra_in1k.bin"
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            m, u = backbone.load_state_dict(state_dict, strict=False)
            print("Load mae state dict from ", checkpoint_path)
            print("Missing key: ", len(m), " Unexpect key: ", len(u))

            num_feats = {'feat': 1280}
            return backbone, num_feats

        elif backbone_name == "maeface":
            backbone = timm.create_model(
                'timm/vit_base_patch16_224.mae',
                pretrained=False,
                num_classes=0,  # remove classifier nn.Linear
                # img_size=224,
            )

            mae_face_path = "/data/zhangfengyu/ABAWcodes/mycode/model/mae_face_pretrain_vit_base.pth"
            mae_face_sd = torch.load(mae_face_path, map_location="cpu")["model"]
            m, u = backbone.load_state_dict(mae_face_sd, strict=False)
            print("Load mae state dict from ", mae_face_path)
            print("Missing key: ", len(m), " Unexpect key: ", len(u))

            num_feats = {'feat': 768}

            return backbone, num_feats
        elif backbone_name == "efficientnetv2":
            backbone = timm.create_model(
                # "timm/efficientnetv2_rw_s.ra2_in1k",
                "timm/efficientnetv2_rw_t.ra2_in1k",
                pretrained=False,
                num_classes=0,  # remove classifier nn.Linear
            )
            # checkpoint_path="/data/zhangfengyu/models/efficientnetv2_rw_s.ra2_in1k.bin"
            checkpoint_path="/data/zhangfengyu/models/efficientnetv2_rw_t.ra2_in1k.bin"
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            m, u = backbone.load_state_dict(state_dict, strict=False)
            print("Load mae state dict from ", checkpoint_path)
            print("Missing key: ", len(m), " Unexpect key: ", len(u))

            # num_feats = {'feat': 1792}
            num_feats = {'feat': 1024}
            return backbone, num_feats

        
        else:
            raise ValueError('Only support regnet at this time.')

    def _reset_parameters(self) -> None:
        # Performs ResNet-style weight initialization
        for m_name, m in self.named_modules():
            if 'backbone' in m_name:
                continue
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def __init__(self):
        # TODO: Load backbone pretrained on static frames
        super(ABAW3Model, self).__init__()
        self.seq_len = cfg.DATA_LOADER.SEQ_LEN
        self.task = cfg.TASK
        self.scale_factor = 1.
        self.threshold = 0.5
        self.learning_rate = cfg.OPTIM.BASE_LR
        self.backbone_name = cfg.MODEL.BACKBONE
        self.backbone_freeze = cfg.MODEL.BACKBONE_FREEZE
        #self.model_name = model_name
        if self.task == 'EXPR':
            # Classification
            self.num_outputs = 8
            self.label_smoothing = cfg.TRAIN.LABEL_SMOOTHING
            # Class weights
            self.cls_weights = nn.Parameter(torch.tensor(
                [0.42715146, 5.79871879, 6.67582676, 4.19317243, 1.01682121, 1.38816715, 2.87961987, 0.32818288],
                requires_grad=False), requires_grad=False) if cfg.TRAIN.LOSS_WEIGHTS else None
            self.loss_func = partial(CEFocalLoss, scale_factor=self.scale_factor, num_classes=self.num_outputs,
                                     label_smoothing=self.label_smoothing,
                                     alpha=cfg.OPTIM.FOCAL_ALPHA, gamma=cfg.OPTIM.FOCAL_GAMMA)

            self.train_metric = F1Score(task='multiclass', threshold=self.threshold, num_classes=self.num_outputs, average='macro')
            self.val_metric = F1Score(task='multiclass', threshold=self.threshold, num_classes=self.num_outputs, average='macro')
        else:
            raise ValueError('Do not know {}'.format(self.task))

        # Dictionary with keys: feats
        self.backbone, self.num_feats = self.get_backbone()

        # if "mobilenetv3" in cfg.MODEL.BACKBONE:
            # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # self.transfer_dim = nn.Sequential(nn.Linear(self.num_feats['feat'], 888), nn.ReLU())

        if 'vggface' in cfg.MODEL.BACKBONE:
            self.down_fc = nn.Sequential(nn.Linear(self.num_feats['feat'], 512), nn.ReLU())
            self.num_feats['feat'] = 512
        else:
            self.down_fc = None

        # if "maeface" in cfg.MODEL.BACKBONE:
            # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # self.transfer_dim = nn.Sequential(nn.Linear(self.num_feats['feat'], 888), nn.ReLU())

        self.fc_aux = None
        # self._reset_parameters()

        if cfg.MODEL.BACKBONE_PRETRAINED != '':
            print('Load pretrained model on static images')
            pretrained_static = torch.load(cfg.MODEL.BACKBONE_PRETRAINED)['state_dict']
            cur_state_dict = self.state_dict()
            for ky in cur_state_dict.keys():
                if 'backbone' in ky and ky in pretrained_static.keys():
                    cur_state_dict[ky] = pretrained_static[ky]
            self.state_dict().update(cur_state_dict)

            pass

        #self.att = Attention(dim = 888, heads = 2, dim_head = 64, dropout = 0.)
        #self.trans = Transformer(dim = 888, depth = 1, heads = 2, dim_head = 64, mlp_dim = 512, dropout = 0.)
        #self.fc = nn.Linear(888*3,self.num_outputs)
        #self.feed = FeedForward(dim = 888, hidden_dim = 512)
        #self.drop = nn.Dropout(0.5)
        self.model_name = cfg.MODEL_NAME

        # 768 -> fc -> 888
        # TODO: dim -> 768
        feat_dim = self.num_feats["feat"]

        if  self.model_name == "combine":
            self.att = Attention(dim = feat_dim, heads = 2, dim_head = 64, dropout = 0.)
            self.trans = Transformer(dim = feat_dim, depth = 1, heads = 2, dim_head = 64, mlp_dim = 512, dropout = 0.)
            self.fc = nn.Linear(feat_dim*3,self.num_outputs)
            self.feed = FeedForward(dim = feat_dim, hidden_dim = 512)
            self.drop = nn.Dropout(0.5)
        elif self.model_name == "no_att_trans":
            self.fc1 = nn.Linear(feat_dim,self.num_outputs)
            self.drop1 = nn.Dropout(0.5)
        elif self.model_name == "only_att":
            self.att2 = Attention(dim = feat_dim, heads = 2, dim_head = 64, dropout = 0.)
            self.fc2 = nn.Linear(feat_dim*2,self.num_outputs)
            self.drop2 = nn.Dropout(0.5)
        elif self.model_name == "only_trans":
            self.trans3 = Transformer(dim = feat_dim, depth = 1, heads = 2, dim_head = 64, mlp_dim = 512, dropout = 0.)
            self.fc3 = nn.Linear(feat_dim*2,self.num_outputs)
            self.drop3 = nn.Dropout(0.5)

    def forward(self, batch):
        #16, 64, 3, 112, 112
        image = batch['image']  # batch size x seq x 3 x h x w
        # Convert to batch size * seq x 3 x h x w for feature extraction
        num_seq = image.shape[0]
        #1024, 3, 112, 112
        feat = torch.reshape(image, (num_seq * self.seq_len,) + image.shape[2:])

        # if self.backbone_name == "samvit" or self.backbone_name == "mobilenetv3":
        #     feat = self.backbone(feat)
        #     feat = self.avg_pool(feat)
        #     feat = torch.reshape(feat, (num_seq, self.seq_len, -1)) # 32, 64, 1280
        #     feat = self.transfer_dim(feat)
        # elif self.backbone_name == "maeface":
        #     feat = self.backbone(feat)  # bxf, 768
        #     feat = self.transfer_dim(feat)  # bxf, 888
        #     feat = torch.reshape(feat, (num_seq, self.seq_len, -1))
        # else:
        try:
            feat = self.backbone(feat)['feat']
        except:
            feat = self.backbone(feat)
        feat = torch.reshape(feat, (num_seq, self.seq_len, -1)) #32, 64, 888

        if  self.model_name == "combine":
            trans = self.trans(feat)
            att = self.att(feat)
            out = torch.cat((feat, att, trans),2)
            out = self.drop(out)
            out = self.fc(out)
        elif self.model_name == "no_att_trans":
            out = self.drop1(feat)
            out = self.fc1(out)
        elif self.model_name == "only_att":
            att = self.att2(feat)
            out = torch.cat((feat, att),2)
            out = self.drop2(out)
            out = self.fc2(out)
        elif self.model_name == "only_trans":
            trans = self.trans3(feat)
            out = torch.cat((feat, trans),2)
            out = self.drop3(out)
            out = self.fc3(out)

        #trans = self.trans(feat)
        #att = self.att(feat)

        #out = torch.cat((feat, att, trans),2)
        #out = self.drop(out)
        #out = self.fc(out)

        if self.down_fc is not None:
            feat = self.down_fc(feat)
        if self.fc_aux is not None:
            out_aux = self.fc_aux(feat)
        else:
            out_aux = None


        return out, out_aux

    def _shared_eval(self, batch, batch_idx, cal_loss=False):
        out, out_aux = self(batch)

        loss = None
        loss_aux_coeff = 0.2
        if cal_loss:
            if self.task != 'MTL':
                loss = self.loss_func(out, batch[self.task])
                if out_aux is not None:
                    loss = loss_aux_coeff * self.loss_func(out, batch[self.task]) + (1 - loss_aux_coeff) * loss

        return out, loss

    def update_metric(self, out, y, is_train=True):
        if self.task == 'EXPR':
            y = torch.reshape(y, (-1,))
            # out = F.softmax(out, dim=1)
        elif self.task == 'AU':
            out = torch.sigmoid(out)
            y = torch.reshape(y, (-1, self.num_outputs))

        elif self.task == 'VA':
            y = torch.reshape(y, (-1, self.num_outputs))

        out = torch.reshape(out, (-1, self.num_outputs))

        if is_train:
            self.train_metric(out, y)
        else:
            self.val_metric(out, y)

    def training_step(self, batch, batch_idx):

        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=True)

        self.log('train_metric', self.train_metric, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=cfg.TRAIN.BATCH_SIZE)

        return loss

    def validation_step(self, batch, batch_idx):
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=False)

        self.log_dict({'val_metric': self.val_metric, 'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True,
                      batch_size=cfg.TEST.BATCH_SIZE)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=False)

        if self.task != 'MTL':
            if self.task == 'EXPR':
                out = torch.argmax(F.softmax(out, dim=-1), dim=-1)
            elif self.task == 'AU':
                out = torch.sigmoid(out)

            return out, batch[self.task], batch['index'], batch['video_id']
        else:
            raise ValueError('Do not implement MTL task.')

    def test_step(self, batch, batch_idx):
        # Copy from validation step
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=False)

        self.log_dict({'test_metric': self.val_metric, 'test_loss': loss}, on_step=False, on_epoch=True, prog_bar=True,
                      batch_size=cfg.TEST.BATCH_SIZE)

    def configure_optimizers(self):

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_training_steps = 336
        if cfg.OPTIM.NAME == 'adam':
            print('Adam optimization ', self.learning_rate)
            opt = torch.optim.Adam(model_parameters, lr=self.learning_rate, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        elif cfg.OPTIM.NAME == 'adamw':
            print('AdamW optimization ', self.learning_rate)
            opt = torch.optim.AdamW(model_parameters, lr=self.learning_rate, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        else:
            print('SGD optimization ', self.learning_rate)
            opt = torch.optim.SGD(model_parameters, lr=self.learning_rate, momentum=cfg.OPTIM.MOMENTUM,
                                  dampening=cfg.OPTIM.DAMPENING, weight_decay=cfg.OPTIM.WEIGHT_DECAY)

        opt_lr_dict = {'optimizer': opt}
        lr_policy = cfg.OPTIM.LR_POLICY
        if lr_policy == 'cos':
            warmup_start_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            scheduler = pl_lr_scheduler.LinearWarmupCosineAnnealingLR(opt, warmup_epochs=cfg.OPTIM.WARMUP_EPOCHS,
                                                                      max_epochs=cfg.OPTIM.MAX_EPOCH,
                                                                      warmup_start_lr=warmup_start_lr,
                                                                      eta_min=cfg.OPTIM.MIN_LR)
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'name': 'lr_sched'}})

        elif lr_policy == 'cos-restart':
            min_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            t_0 = cfg.OPTIM.WARMUP_EPOCHS * self.num_training_steps
            print('Number of training steps: ', t_0 // cfg.OPTIM.WARMUP_EPOCHS)
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=t_0, T_mult=2,
                                                                 eta_min=min_lr)
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})

        elif lr_policy == 'cyclic':
            base_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            step_size_up = self.num_training_steps * cfg.OPTIM.WARMUP_EPOCHS // 2
            mode = 'triangular'  # triangular, triangular2, exp_range
            scheduler = lr_scheduler.CyclicLR(opt, base_lr=base_lr, max_lr=self.learning_rate,
                                              step_size_up=step_size_up, mode=mode, gamma=1., cycle_momentum=False)
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})

        elif lr_policy == 'reducelrMetric':
            scheduler = lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=10, min_lr=1e-7, mode='max')
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'name': 'lr_sched',
                                                 "monitor": "val_metric"}})
        else:
            # TODO: add 'exp', 'lin', 'steps' lr scheduler
            pass
        return opt_lr_dict

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass
######################################################################################################################



