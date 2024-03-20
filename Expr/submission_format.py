import os

from tqdm import tqdm

def convert_submmision_format(input_dir, name, release_txt):

    with open(release_txt, "r") as f:
        file_names = f.readlines()
    file_names = [file_name.strip() for file_name in file_names]

    file_paths = [os.path.join(input_dir, file_name + ".txt") for file_name in file_names]

    results = ["image_location,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other"]

    for file_path, file_name in tqdm(zip(file_paths, file_names), total=len(file_names)):
        with open(file_path, "r") as f:
            result = f.readlines()
            for i, value in enumerate(result[1:], start=1):
                item = f"{file_name}/{i:05d}.jpg,{value.strip()}"
                results.append(item)

    result_path = f"{input_dir}/0_prediction_{name}.txt"
    with open(result_path, "w") as f:
        for i, result in tqdm(enumerate(results), total=len(results)):
            if i == len(results):
                f.write(result)
            else:
                f.write(result+"\n")


if __name__ == "__main__":
    input_dir = "log/EXPR/2024-03-17_17-46-02/predictions"
    result_name = "efficientnetv2-lr0.0005"
    release_txt = "/data/zhangfengyu/ABAW/data/Aff-Wild2/testset/EXPR_test_set_release.txt"

    convert_submmision_format(input_dir, result_name, release_txt)