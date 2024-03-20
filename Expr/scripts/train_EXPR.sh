#!/bin/bash

trap "exit" INT
# shellcheck disable=SC2068

run_ver='v1'
task='EXPR'
logger='wandb'
max_epoch=20
optim_name='adam'
lr_policy='reducelrMetric'
warmup_epoch=1
num_enc_dnc=3
tranf_nhead=8
tranf_dim_fc=1024
model_backbone='regnet-400mf'
#backbone_freeze=('block4' 'block3' 'block2')
backbone_pretrained=""
freeze_bn=True
wd=5e-5
for focal_gamma in 2.0 5.0 7.0 # 9.0
do
  for focal_alpha in 0.25 0.5 0.75 0.9
  do
    for base_lr in 0.0005 0.001 # 0.0001 0.005
    do
      train_dir='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/'$task'_'$run_ver'/'
      python -W ignore main.py --cfg conf/EXPR_baseline.yaml \
              TASK $task \
              LOGGER $logger \
              OUT_DIR $train_dir \
              OPTIM.MAX_EPOCH $max_epoch \
              OPTIM.BASE_LR $base_lr \
              OPTIM.NAME $optim_name \
              OPTIM.LR_POLICY $lr_policy \
              OPTIM.WEIGHT_DECAY $wd \
              OPTIM.FOCAL_ALPHA $focal_alpha \
              OPTIM.FOCAL_GAMMA $focal_gamma \
              OPTIM.WARMUP_EPOCHS $warmup_epoch \
              TRANF.NUM_ENC_DEC $num_enc_dnc \
              TRANF.NHEAD $tranf_nhead \
              TRANF.DIM_FC $tranf_dim_fc \
              MODEL.BACKBONE $model_backbone \
              MODEL.BACKBONE_FREEZE "'block4', 'block3', 'block2'" \
              MODEL.FREEZE_BATCHNORM $freeze_bn \
#              MODEL.BACKBONE_PRETRAINED $backbone_pretrained
      sleep 5
    done
  done
done