import os
import numpy as np
import tqdm
from scipy import interpolate

audio_path = '/data/wenzhuofan/Data/ABAW/Feature/features/wavlm-large-FRA'
video_path = '/data/wenzhuofan/Data/ABAW/Feature/features/clip-vit-large-patch14-FRA'
save_path = '/data/wenzhuofan/Data/ABAW/Feature/features/wavlm-large-CLIPPooling-FRA'

# 获取音频和视频特征文件列表
audio_files = sorted([f for f in os.listdir(audio_path) if f.endswith('.npy')])
video_files = sorted([f for f in os.listdir(video_path) if f.endswith('.npy')])

# 遍历音频和视频特征文件，进行对齐
for audio_file, video_file in tqdm.tqdm(zip(audio_files, video_files), total=len(audio_files)):
    # 读取音频和视频特征数据
    audio_data = np.load(os.path.join(audio_path, audio_file))
    audio_data = 2 * (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data)) - 1
    video_data = np.load(os.path.join(video_path, video_file))

    # 对齐音频特征到视频特征长度
    if audio_data.shape[0] != video_data.shape[0]:
        if audio_data.shape[0] < video_data.shape[0]:
            # 如果音频特征较短，采用插值方式填充
            x_old = np.arange(audio_data.shape[0])
            x_new = np.linspace(0, audio_data.shape[0] - 1, video_data.shape[0])
            interpolator = interpolate.interp1d(x_old, audio_data, axis=0, kind='linear')
            audio_data = interpolator(x_new)
        else:
            # 如果音频特征较长，采用池化方式降采样
            audio_length = audio_data.shape[0]
            video_length = video_data.shape[0]
            pool_factor = audio_length / video_length
            pool_factor_continuous = np.linspace(0, audio_length - 1, video_length)
            pool_factor_continuous = np.round(pool_factor_continuous).astype(int)
            audio_data = audio_data[pool_factor_continuous]

    # 打印对齐后的特征数据形状
    # print('audio',audio_data.shape,'video',video_data.shape)
    np.save(os.path.join(save_path, audio_file), audio_data)

    # 在这里进行你的其他操作，比如处理对齐后的特征数据或进行其他计算等
