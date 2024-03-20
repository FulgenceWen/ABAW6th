import os

new_label_save_root='/data/wenzhuofan/Data/ABAW/Split_Feature_Label/split_label/Train_Set/o_800s_1000'
feature_path='/data/wenzhuofan/Data/ABAW/Split_Feature_Label/split_feature/wavlm-large-CLIPPooling-o_800s_1000-FRA'
#
# cnt=0
# for file in sorted(os.listdir(new_label_save_root)):
#     if file[:-4]+'.npy' not in os.listdir(feature_path):
#         print(file)
#         cnt += 1
#
# print(cnt)

l= list(range(0, 100))

print(l[20:10])