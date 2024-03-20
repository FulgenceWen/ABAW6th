import os

denosise_audio='/data/wenzhuofan/Data/ABAW/Aff-Wild2/raw/Raw_audio/denoise'
fixed_video_path='/data/wenzhuofan/Data/ABAW/Aff-Wild2/cropped-aligned/CA_all'

print('file not in fixed_video_path')
for file in os.listdir(denosise_audio):
    if file[:-4] not in os.listdir(fixed_video_path):
        os.remove(os.path.join(denosise_audio,file))


existing_files_with_extension = os.listdir(denosise_audio)  # 例如，这里是你已知的具有后缀的文件列表

# 你的文件名列表（缺少后缀）
file_names_without_extension = os.listdir(fixed_video_path)  # 例如，这里是你的文件名列表

for i in range(len(existing_files_with_extension)):
    existing_files_with_extension[i] = existing_files_with_extension[i][:-4]
# 打印缺少后缀的文件名
print("Missing files with extension:")
for file in file_names_without_extension:
    if file not in existing_files_with_extension:
        print(file)