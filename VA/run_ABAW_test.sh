cd feature_extraction/audio
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=4 extract_audio_huggingface.py --dataset=ABAW --feature_level=FRAME --model_name=wavlm-large --gpu=0