import os
from tqdm import tqdm


if __name__ == "__main__":

    our_file = "log/EXPR/2024-03-17_17-46-02/predictions/0_prediction_efficientnetv2-lr0.0005.txt"
    our_i = 0

    fix_our_file = f"{our_file.strip('.txt')}-fix.txt"
    fix_list = []

    with open(our_file, "r") as f:
        our_results = f.readlines()

    ref_file = "predictions.txt"
    with open(ref_file, "r") as f:
        ref_results = f.readlines()

    print("our ", len(our_results))
    print("ref ", len(ref_results))

    for i, ref_result in enumerate(tqdm(ref_results, total=len(ref_results))):

        if ref_results[i].split(',')[0] == our_results[our_i].split(',')[0]:
            fix_list.append(our_results[our_i])
            our_i += 1
        else:
            print("No ", ref_results[i])
            path = ref_result.split(',')[0]
            label = our_results[our_i].split(',')[-1]
            item = f"{path},{label}"
            print("Correct ", item)
            fix_list.append(item)

        # print(our_results[fix_index].split(',')[0], ref_result.split(',')[0])
        # if fix_index < len(our_results) and our_results[fix_index].split(',')[0] == ref_result.split(',')[0]:
            # fix_list.append(our_results[fix_index])
            # fix_index += 1
        # else:
            # print("No ", ref_result)
            # fix_list.append(ref_result)

        # if ref_result.split(",")[0] != our_results[i].split(",")[0]:
        #     label = our_results[i].split(",")[-1]
        #     name = our_results[i].split("/")[0]
        #     id = int(our_results[i].split(',')[0].split('/')[-1].strip(".jpg")) + 1
        #     item = f"{name}/{id:05d}.jpg,{label}"
        #     our_results.insert(i-1, item)
    
    print("correct ", len(fix_list))

    with open(fix_our_file, "w") as f:
        for i, result in tqdm(enumerate(fix_list), total=len(fix_list)):
            f.write(result)
    print("Saving ", fix_our_file)
        

