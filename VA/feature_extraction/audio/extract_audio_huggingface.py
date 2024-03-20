import os
import math
import time
import glob
import torch
import argparse
import numpy as np
import soundfile as sf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# import config
import sys
sys.path.append('../../')
import config

from transformers import AutoModel
from transformers import WhisperFeatureExtractor, Wav2Vec2FeatureExtractor


# supported models
################## ENGLISH ######################
WAV2VEC2_BASE = 'wav2vec2-base-960h' # https://huggingface.co/facebook/wav2vec2-base-960h
WAV2VEC2_LARGE = 'wav2vec2-large-960h' # https://huggingface.co/facebook/wav2vec2-large-960h
DATA2VEC_AUDIO_BASE = 'data2vec-audio-base-960h' # https://huggingface.co/facebook/data2vec-audio-base-960h
DATA2VEC_AUDIO_LARGE = 'data2vec-audio-large' # https://huggingface.co/facebook/data2vec-audio-large

################## CHINESE ######################
HUBERT_BASE_CHINESE = 'chinese-hubert-base' # https://huggingface.co/TencentGameMate/chinese-hubert-base
HUBERT_LARGE_CHINESE = 'chinese-hubert-large' # https://huggingface.co/TencentGameMate/chinese-hubert-large
WAV2VEC2_BASE_CHINESE = 'chinese-wav2vec2-base' # https://huggingface.co/TencentGameMate/chinese-wav2vec2-base
WAV2VEC2_LARGE_CHINESE = 'chinese-wav2vec2-large' # https://huggingface.co/TencentGameMate/chinese-wav2vec2-large

################## Multilingual #################
WAVLM_BASE = 'wavlm-base' # https://huggingface.co/microsoft/wavlm-base
WAVLM_LARGE = 'wavlm-large' # https://huggingface.co/microsoft/wavlm-large
WHISPER_BASE = 'whisper-base' # https://huggingface.co/openai/whisper-base
WHISPER_LARGE = 'whisper-large-v3' # https://huggingface.co/openai/whisper-large-v2

## Target: avoid too long inputs
# input_values: [1, wavlen], output: [bsize, maxlen]
def split_into_batch(input_values, maxlen=16000*10):
    if len(input_values[0]) <= maxlen:
        return input_values
    
    bs, wavlen = input_values.shape
    assert bs == 1
    tgtlen = math.ceil(wavlen / maxlen) * maxlen
    batches = torch.zeros((1, tgtlen))
    batches[:, :wavlen] = input_values
    batches = batches.view(-1, maxlen)
    return batches

def extract(model_name, audio_files, save_dir, feature_level, gpu,local_rank=-1):

    start_time = time.time()

    # load model
    model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')
    
    if model_name in [WHISPER_BASE, WHISPER_LARGE]:
        model = AutoModel.from_pretrained(model_file)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_file)
    else:
        model = AutoModel.from_pretrained(model_file)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)

    if gpu != -1 and args.local_rank == -1:
        device = torch.device(f'cuda:{gpu}')
        model.to(device)


    if args.local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')

        model.to(device)
        model = DDP(model, device_ids=[local_rank],output_device=local_rank)

    model.eval()

    # iterate audios
    for idx, audio_file in enumerate(audio_files, 1):
        file_name = os.path.basename(audio_file)
        vid = file_name[:-4]
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')

        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if os.path.exists(csv_file):
            print(f'"{csv_file}" already exists, skip...')
            continue

        ## process for too short ones
        samples, sr = sf.read(audio_file)
        assert sr == 16000, 'currently, we only test on 16k audio'
        
        ## model inference
        with torch.no_grad():
            if model_name in [WHISPER_BASE, WHISPER_LARGE]:
                layer_ids = [-1]
                input_features = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_features # [1, 80, 3000]
                decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
                if gpu != -1: input_features = input_features.to(device)
                if gpu != -1: decoder_input_ids = decoder_input_ids.to(device)
                last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
                assert last_hidden_state.shape[0] == 1
                feature = last_hidden_state[0].detach().squeeze().cpu().numpy() # (2, D)
            else:
                layer_ids = [-4, -3, -2, -1]
                input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values # [1, wavlen]
                input_values = split_into_batch(input_values) # [bsize, maxlen=10*16000]
                hidden_states_list = []
                batch_size = 4  # 定义每个批次的大小
                for i in range(0, len(input_values), batch_size):
                    batch_input = input_values[i:i + batch_size]
                    if gpu != -1:
                        batch_input = batch_input.to(device)
                    hidden_states = model(batch_input, output_hidden_states=True).hidden_states  # tuple of (B, T, D)
                    hidden_states_list.append(hidden_states)

                # hidden_states_concat = [torch.cat([h[idx] for h in hidden_states_list], dim=0) for idx in range(len(hidden_states_list[0]))]  # 拼接隐含层结果
                num_layers = len(hidden_states_list[0])

                # 创建一个空列表来存储拼接后的结果
                hidden_states_concat = []

                # 遍历张量列表的索引范围
                for idx in range(-4,0):
                    # 创建一个临时列表来存储当前索引位置的张量
                    tensors_at_idx = []
                    # 遍历 hidden_states_list 中的张量列表
                    for hidden_states in hidden_states_list:
                        # 将当前索引位置的张量添加到临时列表中
                        tensors_at_idx.append(hidden_states[idx])
                    # 在指定维度上拼接临时列表中的张量，并将结果添加到结果列表中
                    concatenated_tensor = torch.cat(tensors_at_idx, dim=0)
                    hidden_states_concat.append(concatenated_tensor)

                # feature = torch.stack(hidden_states_concat)[layer_ids].sum(dim=0)  # (B, T, D) # -> compress waveform channel

                # 首先创建一个空列表来存储连接后的结果
                concatenated_tensors = []

                # 按层连接隐藏状态张量
                layer_tensors = [hidden_states_concat[layer_id] for layer_id in layer_ids]
                layer_tensors = torch.stack(layer_tensors)  # (num_layers, B, T, D)

                feature = layer_tensors.sum(dim=0)

                bsize, segnum, featdim = feature.shape
                feature = feature.view(-1, featdim).detach().squeeze().cpu().numpy() # (B*T, D)

        ## store values
        if feature_level == 'UTTERANCE':
            feature = np.array(feature).squeeze()
            if len(feature.shape) != 1:
                feature = np.mean(feature, axis=0)
            np.save(csv_file, feature)
        else:
            np.save(csv_file, feature)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu')
    parser.add_argument('--model_name', type=str, default='chinese-hubert-large', help='feature extractor')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='FRAME or UTTERANCE')
    parser.add_argument('--dataset', type=str, default='MER2023', help='input dataset')
    # ------ 临时测试SNR对于结果的影响 ------
    parser.add_argument('--noise_case', type=str, default=None, help='extract feature of different noise conditions')
    # ------ 临时测试 tts audio 对于结果的影响 -------
    parser.add_argument('--tts_lang', type=str, default=None, help='extract feature from tts audio, [chinese, english]')
    parser.add_argument("--local_rank",default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()

    # analyze input
    audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    save_dir = config.PATH_TO_FEATURES[args.dataset]
    if args.noise_case is not None:
        audio_dir += '_' + args.noise_case
    if args.tts_lang is not None:
        audio_dir += '-' + f'tts{args.tts_lang[:3]}16k'

    # audio_files
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')

    # save_dir
    if args.noise_case is not None:
        dir_name = f'{args.model_name}-noise{args.noise_case}-{args.feature_level[:3]}'
    elif args.tts_lang is not None:
        dir_name = f'{args.model_name}-tts{args.tts_lang[:3]}-{args.feature_level[:3]}'
    else:
        dir_name = f'{args.model_name}-{args.feature_level[:3]}'

    save_dir = os.path.join('/data/wenzhuofan/Data/ABAW/Feature/features', dir_name)
    # save_dir = os.path.join('save_dir', dir_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # extract features
    extract(args.model_name, audio_files, save_dir, args.feature_level,gpu=args.gpu,local_rank=args.local_rank)

