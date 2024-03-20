import os
import shutil
import pickle
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *
from toolkit.utils.read_files import *

emos = ['A', 'C', 'D', 'F', 'H', 'N', 'S', 'U']
emo2idx, idx2emo = {}, {}
for ii, emo in enumerate(emos):
    emo2idx[emo] = ii
    idx2emo[ii]  = emo

def map_label_to_index(label):
    return emo2idx[label]
def load_transcription(trans_path, save_path):
    ## read pkl file
    names, sentences = [], []
    file_names = os.listdir(trans_path)
    for name in file_names:
        names.append(name.split('.')[0])
        sentences.append(open(os.path.join(trans_path,name), 'r').read())
    print(f'whole sample number: {len(names)}')

    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [sentences[ii]]
    func_write_key_to_csv(save_path, names, name2key, ['english'])


def read_train_val_test(label_path, data_type):
    names, labels = [], []
    df= pd.read_csv(label_path)
    df=df[(df['EmoClass'] != 'X') & (df['EmoClass'] != 'O')]
    df_row=None
    assert data_type in ['train', 'val', 'test']
    if data_type == 'train': df_row = df[df['Split_Set'] == 'Train']
    if data_type == 'val':   df_row = df[df['Split_Set'] == 'Development']
    if data_type == 'test':  df_row = df[df['Split_Set'] == 'Test']
    names=df_row['FileName'].tolist()
    df_row['IntegerLabel'] = df['EmoClass'].map(map_label_to_index)
    labels=df_row['IntegerLabel'].tolist()
    names=[i.split('.')[0] for i in names]
    return names, labels


def normalize_dataset_format(data_root, save_root):
    # gain paths
    label_path = os.path.join(data_root,'Labels', 'labels_consensus_withTest.csv')
    transcript_path = os.path.join(data_root, 'Transcripts')
    assert os.path.exists(label_path), f'must has a pre-processed label file'
    # video_root = os.path.join(data_root, 'Video/Segmented')

    # gain (names, labels)
    train_names, train_labels = read_train_val_test(label_path, 'train')
    val_names, val_labels = read_train_val_test(label_path, 'val')
    test_names, test_labels = read_train_val_test(label_path, 'test')
    print(f'train number: {len(train_names)}')
    print(f'val   number: {len(val_names)}')
    print(f'test  number: {len(test_names)}')

    ## output path
    save_video = os.path.join(save_root, 'subvideo')
    save_label = os.path.join(save_root, 'label.npz')
    save_trans = os.path.join(save_root, 'transcription.csv')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    # if not os.path.exists(save_video): os.makedirs(save_video)

    ## load transcripts
    load_transcription(transcript_path, save_trans)

    ## generate label path
    whole_corpus = {}
    for name, videonames, labels in [('train', train_names, train_labels),
                                     ('val', val_names, val_labels),
                                     ('test', test_names, test_labels)]:
        whole_corpus[name] = {}
        for ii, videoname in enumerate(videonames):
            whole_corpus[name][videoname] = {'emo': labels[ii], 'val': -10}

    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        val_corpus=whole_corpus['val'],
                        test_corpus=whole_corpus['test'])


if __name__ == '__main__':
    data_root = '/data/wenzhuofan/Data/Odyssey'
    save_root = '/data/wenzhuofan/Data/Odyssey/Labels/odyssey-process'
    normalize_dataset_format(data_root, save_root)

    # data_root = 'H:\\desktop\\Multimedia-Transformer\\chinese-mer-2023\\dataset\\cmumosi-process'
    # trans_path = os.path.join(data_root, 'transcription.csv')
    # polish_path = os.path.join(data_root, 'transcription-engchi-polish.csv')
    # func_translate_transcript_polish_merge(trans_path, polish_path) # 再次检测一下遗漏的部分
