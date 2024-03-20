import os
import shutil

fixed_video_path='/data/wenzhuofan/Data/ABAW/Aff-Wild2/cropped-aligned/CA_all'
audio_path="/data/wenzhuofan/Data/ABAW/Aff-Wild2/raw/Raw_audio/denoise"
existing_files_with_extension = os.listdir(audio_path)  # 例如，这里是你已知的具有后缀的文件列表

# 你的文件名列表（缺少后缀）
file_names_without_extension = os.listdir(fixed_video_path)  # 例如，这里是你的文件名列表

for i in range(len(existing_files_with_extension)):
    existing_files_with_extension[i] = existing_files_with_extension[i][:-4]
# 打印缺少后缀的文件名
for file in file_names_without_extension:
    if file not in existing_files_with_extension:
        real_name='_'.join(file.split('_')[:-1])
        shutil.copy(os.path.join(audio_path,real_name + ".wav"), os.path.join(audio_path,file+".wav"))

print("Done")