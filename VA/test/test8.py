# 读取文本文件
import os
preds_path='/data/wenzhuofan/Data/ABAW/Aff-Wild2/ABAW_Annotations/VA_Estimation_Challenge/Pesudo_Test'
preds_file=os.path.join('/data/wenzhuofan/Data/ABAW/Aff-Wild2/ABAW_Annotations/VA_Estimation_Challenge/Pesudo_Test',
                        'test1_features:clip-vit-large-patch14-o_800s_1000-FRA_dataset:ABAW_model:mctn+frm_align+None_20240317_082440.txt')

with open(preds_file, "r") as file:
    lines = file.readlines()
lines = lines[1:]
# 分割内容并创建新文件
current_filename = None
file_content = []

for line in lines:
    if line.strip():  # 忽略空行
        image_location, valence, arousal = line.strip().split(",")
        filename = image_location.split("/")[0]

        if filename != current_filename:
            # 写入之前文件的内容
            if file_content:
                with open(os.path.join(preds_path,f"{current_filename}.txt"), "w") as output_file:
                    output_file.write("valence,arousal\n")
                    output_file.writelines(file_content)
            # 重置当前文件内容和文件名
            current_filename = filename
            file_content = []

        # 将valence和arousal写入文件内容
        file_content.append(f"{valence},{arousal}\n")

# 写入最后一个文件的内容
if file_content:
    with open(os.path.join(preds_path,f"{current_filename}.txt"), "w") as output_file:
        output_file.write("valence,arousal\n")
        output_file.writelines(file_content)