import os

import numpy as np
import tqdm
import numpy

audio='/data/wenzhuofan/Data/ABAW/Split_Feature_Label/split_feature/wavlm-large-CLIPPooling-o_800s_1000-FRA'
video='/data/wenzhuofan/Data/ABAW/Feature/features/clip-vit-large-patch14-FRA'

a = np.load(os.path.join(video,'102-30-640x360.npy'))
print(a.shape)
# audio_length=[]
# audio_name=[]
# video_length=[]
# video_name=[]
# cnt=0
# for a,v in tqdm.tqdm(zip(sorted(os.listdir(audio)),sorted(os.listdir(video)))):
#     if a.split('_')[-1]=='right.npy' or a.split('_')[-1]=='left.npy':
#         continue
#     print(a,v)
#     data=np.load(os.path.join(audio,a))
#     audio_length.append(data.shape[0])
#     data2=np.load(os.path.join(video,v))
#     video_length.append(data2.shape[0])
#     audio_name.append(a)
#     video_name.append(v)
# #
# diff=[abs(audio_length[i]/video_length[i]) for i in range(len(audio_length))]
# avg_diff=abs(np.mean(audio_length)/np.mean(video_length))
# print(avg_diff)
# max_diff = np.max(diff)
# min_diff = np.min(diff)
# print(diff.index(max_diff))
#
# print(audio_name[diff.index(max_diff)])
# print(video_name[diff.index(max_diff)])
# print(audio_length[diff.index(max_diff)])
# print(video_length[diff.index(max_diff)])
#
# # 打印最大值和最小值
# print("最大差距：", max_diff)
# print("最小差距：", min_diff)
