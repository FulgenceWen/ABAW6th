import os
import re

file_path='/data/wenzhuofan/Data/ABAW/Aff-Wild2/raw/Raw_audio/denoise'
# 使用正则表达式替换以数字加下划线开头的部分和_(Vocals)
for file in sorted(os.listdir(file_path)):
    filename = file
    filename_without_id_vocals = re.sub(r'^\d+_|_\(Vocals\)', '', filename)
    os.rename(os.path.join(file_path, file), os.path.join(file_path, filename_without_id_vocals))
    print(filename_without_id_vocals)