import os
import shutil
import pickle

import numpy as np

from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *
from toolkit.utils.read_files import *
import os
import glob
import tqdm
import pickle

import config
from toolkit.globals import *
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *
from toolkit.utils.read_files import *


# split short video from long video
def split_feature_by_length_ABAW(data_root, save_root,args):
    ## video number 3837
    #trans_root = os.path.join(data_root, 'Transcript/Segmented/Combined')  # 3837 samples
    test_path='/data/wenzhuofan/Data/ABAW/New_Label/Test_Set'
    test_save_path='/data/wenzhuofan/Data/ABAW/Split_Feature_Label/split_label/Test_Set'
    feature_name = args.feature.split('-')
    if args.pesudo:
        new_feature_name = ('-'.join(feature_name[:-1]) + '-' + 'o_'+str(args.overlap)+'s_'+str(args.split_length)
                            + '-'+'pesudo'+'-' + feature_name[-1])
        save_root = os.path.join(save_root, new_feature_name)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        else:
            print('feature already exists')
            return
        for file in tqdm.tqdm(sorted(os.listdir(os.path.join(data_root, args.feature)))):
            create_label = False
            if file[:-4] + '.txt' in os.listdir(test_path):
                create_label = True
            feature = np.load(os.path.join(data_root, args.feature, file))
            length = feature.shape[0]
            split_length = args.split_length
            overlap = args.overlap
            stride = split_length - overlap  # 步长
            if create_label:
                num_windows = (math.ceil((length - split_length) / split_length) + 1)*(split_length//stride)  # 计算可以获得的窗口数
            else:
                num_windows = (length - split_length) // stride + 1  # 计算可以获得的窗口数

            for i in range(num_windows):
                if create_label and i % (split_length/stride) == 0:
                    label_name = 'o_' + str(args.overlap) + 's_' + str(args.split_length)+'_pesudo'
                    new_label_save_path = os.path.join(test_save_path, label_name)
                    if not os.path.exists(new_label_save_path):
                        os.makedirs(new_label_save_path)
                    label_file = open(os.path.join(new_label_save_path, file[:-4] + '_' + str(i) + ".txt"), "w")
                    label_file.close()
                start = i * stride  # 窗口的起始位置
                end = min(start + split_length, length)  # 窗口的结束位置
                window_feature = feature[start:end]  # 切割特征
                # 保存窗口特征
                np.save(os.path.join(save_root, file[:-4] + '_' + str(i) + '.npy'), window_feature)
    else:
        new_feature_name = '-'.join(feature_name[:-1]) + '-' + 'o_'+str(args.overlap)+'s_'+str(args.split_length) + '-' + feature_name[-1]
        save_root = os.path.join(save_root, new_feature_name)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        else:
            print('feature already exists')
            return
        for file in tqdm.tqdm(sorted(os.listdir(os.path.join(data_root, args.feature)))):
            create_label = False
            if file[:-4]+'.txt' in os.listdir(test_path):
                create_label=True
            feature = np.load(os.path.join(data_root, args.feature, file))
            length = feature.shape[0]
            split_length = args.split_length
            overlap = args.overlap
            stride = split_length - overlap  # 步长
            if create_label:
                num_windows= math.ceil((length - split_length) / split_length) + 1  # 计算可以获得的窗口数
            else:
                num_windows = (length - split_length) // stride + 1  # 计算可以获得的窗口数
            for i in range(num_windows):
                if create_label:
                    start = i * split_length  # 窗口的起始位置
                    end = min(start + split_length, length)  # 窗口的结束位置，取 start + split_length 和 length 的较小值
                    window_feature = feature[start:end]  # 切割特征
                    # 保存窗口特征
                    np.save(os.path.join(save_root, file[:-4] + '_' + str(i) + '.npy'), window_feature)

                    label_name = 'o_' + str(args.overlap) + 's_' + str(args.split_length)
                    new_label_save_path = os.path.join(test_save_path, label_name)
                    if not os.path.exists(new_label_save_path):
                        os.makedirs(new_label_save_path)
                    label_file = open(os.path.join(new_label_save_path, file[:-4]+ '_' + str(i) + ".txt"), "w")
                    label_file.close()
                else:
                    start = i * stride  # 窗口的起始位置
                    end = start + split_length  # 窗口的结束位置
                    window_feature = feature[start:end]  # 切割特征
                    # 保存窗口特征
                    np.save(os.path.join(save_root, file[:-4] + '_' + str(i) + '.npy'), window_feature)

def split_label_by_length_ABAW(data_root, save_root,args):
    if not os.path.exists(os.path.join(save_root,'Train_Set')): os.makedirs(os.path.join(save_root,'Train_Set'))
    if not os.path.exists(os.path.join(save_root,'Validation_Set')): os.makedirs(os.path.join(save_root,'Validation_Set'))
    #if not os.path.exists(os.path.join(save_root,'Test_Set')): os.makedirs(os.path.join(save_root,'Test_Set'))
    for dir in tqdm.tqdm(sorted(os.listdir(data_root))):

        if not args.pesudo and dir=='Pesudo_Train_Set':
            continue
        if args.pesudo and dir=='Train_Set':
            continue
        if dir=='Test_Set': continue

        names = []
        labels = []
        if args.pesudo:
            label_name = 'o_'+str(args.overlap)+'s_'+str(args.split_length)+'_pesudo'
        else:
            label_name = 'o_'+str(args.overlap)+'s_'+str(args.split_length)
        new_label_save_path = os.path.join(save_root, dir, label_name)
        if args.pesudo and (dir=='Pesudo_Train_Set' or dir=='Train_Set'):
            new_label_save_path = os.path.join(save_root, 'Train_Set', label_name)
        if not os.path.exists(new_label_save_path):
            os.makedirs(new_label_save_path)
        else:
            print('label already exists')
            continue
        for file in sorted(os.listdir(os.path.join(data_root, dir))):
            with open(os.path.join(data_root, dir, file), 'r') as f:
                label = f.readlines()
            label = label[1:]
            label = [[float(j) for j in i.split(',')] for i in label]
            length = len(label)
            split_length = args.split_length
            overlap = args.overlap
            stride = split_length - overlap  # 步长
            num_windows = (length - split_length) // stride + 1  # 计算可以获得的窗口数
            for i in range(num_windows):
                start = i * stride  # 窗口的起始位置
                end = start + split_length  # 窗口的结束位置
                window_label = label[start:end]  # 截取label
                # 添加到names和labels列表中
                names.append(file[:-4] + '_' + str(i))
                labels.append(window_label)
        for name,label in zip(names,labels):
            with open(os.path.join(new_label_save_path, name+'.txt'), 'w') as f:
                # write f as valance,arousal\nxx,xx\nxx,xx\n
                f.write('valance,arousal\n')
                for i in label:
                    f.write(str(i[0]) + ',' + str(i[1]) + '\n')
            f.close()



def read_train_val_test(label_path, data_type,args):
    names, labels = [], []
    if args.pesudo:
        label_name = 'o_' + str(args.overlap) + 's_' + str(args.split_length)+'_pesudo'
    else:
        label_name = 'o_' + str(args.overlap) + 's_' + str(args.split_length)
    assert data_type in ['train', 'val', 'test']
    # videoIDs, videoLabels, _, _, trainVids, valVids, testVids = pickle.load(open(label_path, "rb"), encoding='utf-8')
    if data_type == 'train':
        if args.pesudo:
            for file in sorted(os.listdir(os.path.join(label_path, 'Train_Set', label_name))):
                with open(os.path.join(label_path, 'Train_Set', label_name, file), 'r') as f:
                    label = f.readlines()
                label = label[1:]
                label = [[float(j) for j in i.split(',')] for i in label]
                names.append(file[:-4])
                labels.append(label)
        else:
            for file in sorted(os.listdir(os.path.join(label_path, 'Train_Set',label_name))):
                with open(os.path.join(label_path, 'Train_Set',label_name, file), 'r') as f:
                    label = f.readlines()
                label = label[1:]
                label = [[float(j) for j in i.split(',')] for i in label]
                names.append(file[:-4])
                labels.append(label)
    if data_type == 'val':
        for file in sorted(os.listdir(os.path.join(label_path, 'Validation_Set',label_name))):
            with open(os.path.join(label_path, 'Validation_Set',label_name, file), 'r') as f:
                label = f.readlines()
            label = label[1:]
            label = [[float(j) for j in i.split(',')] for i in label]
            names.append(file[:-4])
            labels.append(label)
    if data_type == 'test':
        for file in sorted(os.listdir(os.path.join(label_path, 'Test_Set', label_name))):
            with open(os.path.join(label_path, 'Test_Set', label_name, file), 'r') as f:
                label = f.readlines()
            label = label[1:]
            label = [[float(j) for j in i.split(',')] for i in label]
            names.append(file[:-4])
            labels.append(label)
    return names, labels


def normalize_dataset_format(data_root, save_root,args):
    # gain paths
    feature_path=os.path.join(data_root,'Feature','features')
    label_path = os.path.join(data_root,'New_Label')

    ## output path
    save_split_feature = os.path.join(save_root,'split_feature')
    save_label = os.path.join(save_root,'split_label')
    # save_trans = os.path.join(save_root, 'transcription.csv')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_split_feature): os.makedirs(save_split_feature)
    if not os.path.exists(save_label): os.makedirs(save_label)

    # save videos [TODO: test]
    split_feature_by_length_ABAW(feature_path, save_split_feature,args)
    split_label_by_length_ABAW(label_path, save_label,args)


    # # gain (names, labels)
    train_names, train_labels = read_train_val_test(save_label, 'train',args)
    val_names, val_labels = read_train_val_test(save_label, 'val',args)
    test_names, test_labels = read_train_val_test(save_label, 'test',args)
    if args.shuffle:
        train_data = list(zip(train_names, train_labels))
        val_data = list(zip(val_names, val_labels))
        random.shuffle(train_data)
        random.shuffle(val_data)

        # 计算划分比例
        train_size = int(len(train_data) * 0.75)
        val_size = len(train_data) - train_size

        # 划分测试集和验证集
        train_set = train_data[:train_size]
        val_set = train_data[train_size:]

        # 解压缩样本和标签
        train_names, train_labels = zip(*train_set)
        val_names, val_labels = zip(*val_set)

    print(f'train: {len(train_names)}')
    print(f'val:   {len(val_names)}')
    print(f'test:  {len(test_names)}')

    ## generate label path
    whole_corpus = {}
    for name, videonames, labels in [('train', train_names, train_labels),
                                     ('val', val_names, val_labels),
                                     ('test', test_names, test_labels)]:
        whole_corpus[name] = {}
        print(f'{name}: sample number: {len(videonames)}')
        for ii, videoname in enumerate(videonames):
            whole_corpus[name][videoname] = {'emo': 0, 'val': labels[ii]}

    if args.shuffle:
        random_number = random.randint(1, 1000)  # 生成一个范围在1到1000之间的随机整数
        random_number_str = str(random_number).zfill(4)  # 将随机整数转换为字符串，并在左侧用0填充至总长度为4位
        if args.pesudo:
            np.savez_compressed(os.path.join(save_label, f'label_{random_number_str}_pesudo.npz'),
                            train_corpus=whole_corpus['train'],
                            val_corpus=whole_corpus['val'],
                            test_corpus=whole_corpus['test'])
        else:
            np.savez_compressed(os.path.join(save_label, f'label_{random_number_str}.npz'),
                                train_corpus=whole_corpus['train'],
                                val_corpus=whole_corpus['val'],
                                test_corpus=whole_corpus['test'])
    else:
        np.savez_compressed(os.path.join(save_label, 'label.npz'),
                        train_corpus=whole_corpus['train'],
                        val_corpus=whole_corpus['val'],
                        test_corpus=whole_corpus['test'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_length', type=int, default=200)
    parser.add_argument('--overlap', type=int, default=100)
    parser.add_argument('--feature',type=str, default=None)
    parser.add_argument('--shuffle',action='store_true')
    parser.add_argument('--pesudo',action='store_true')
    args=parser.parse_args()
    if args.pesudo:
        args.shuffle=True
    data_root = '/data/wenzhuofan/Data/ABAW'
    save_root = '/data/wenzhuofan/Data/ABAW/Split_Feature_Label'
    normalize_dataset_format(data_root, save_root,args)
