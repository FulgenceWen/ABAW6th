 cd feature_extraction/visual
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 extract_vision_hf_img.py --dataset=ABAW --feature_level=FRAME --model_name=dino-vitb8
