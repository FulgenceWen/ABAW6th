import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *
from toolkit.data import get_datasets
import audmetric

class ABAW():
    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.label_path = config.PATH_TO_LABEL[args.dataset]
        self.gpu_num=len(args.gpu)

        self.dataset = args.dataset
        assert self.dataset in ['ABAW']

        # update args
        args.output_dim1 = 0
        args.output_dim2 = 2
        args.metric_name = 'ccc'

    def get_loaders(self):
        dataloaders = []
        for data_type in ['train', 'val', 'test']:
            names, labels = self.read_names_labels(self.label_path, data_type, debug=self.debug)
            print(f'{data_type}: sample number {len(names)}')

            if data_type in ['train', 'val']:
                dataset = get_datasets(self.args, names, labels)
            else:
                dataset = get_datasets(self.args, names, labels)

            if data_type in ['train']:
                dataloader = DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        collate_fn=dataset.collater,
                                        pin_memory=True)
            elif data_type in ['val']:
                dataloader = DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        collate_fn=dataset.collater,
                                        shuffle=False,
                                        pin_memory=True)
            else:
                dataloader = DataLoader(dataset,
                                        batch_size=1,
                                        num_workers=self.num_workers,
                                        collate_fn=dataset.collater,
                                        shuffle=False,
                                        pin_memory=True)
            dataloaders.append(dataloader)
        train_loaders = [dataloaders[0]]
        eval_loaders = [dataloaders[1]]
        test_loaders = [dataloaders[2]]

        return train_loaders, eval_loaders, test_loaders

    def read_names_labels(self, label_path, data_type, debug=False):
        names, labels = [], []
        if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
        if data_type == 'val':   corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
        if data_type == 'test':  corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
        for name in corpus:
            names.append(name)
            labels.append(corpus[name])
        # for debug
        if debug:
            names = names[:100]
            labels = labels[:100]
        elif data_type == 'train' or data_type == 'val':
            names= names[:len(names)//self.gpu_num*self.gpu_num]
            labels= labels[:len(labels)//self.gpu_num*self.gpu_num]
        return names, labels

    def CCC_loss(self, preds, labels):
        return audmetric.concordance_cc(preds, labels)

    def CCC_total(self, arousal, valence):
        return (arousal + valence) / 2


    def calculate_results(self, emo_probs=[], emo_labels=[],val_preds=[], val_labels=[]):

        # 将emo_probs和emo_labels转换为tensor，并在有可用的情况下移动到cuda
        # total_pred = torch.tensor(val_preds, dtype=torch.float32).view(-1, 1)
        # total_y = torch.tensor(val_labels, dtype=torch.float32).view(-1, 1)
        #
        # if torch.cuda.is_available():
        #     total_pred = total_pred.cuda()
        #     total_y = total_y.cuda()
        # print(val_preds.shape)
        # print(val_labels.shape)
        arousal_preds = val_preds[:,:,1]
        valence_preds = val_preds[:,:, 0]
        arousal_labels = val_labels[:,:, 1]
        valence_labels = val_labels[:,:, 0]
        batch_size, seq_len= arousal_preds.shape
        if np.isnan(arousal_preds).any() or np.isnan(valence_preds).any():
            print('nan in preds')
        # 使用列表推导式计算每个序列的 CCC 损失
        # print(arousal_preds.shape,arousal_labels.shape)
        arousal_losses = [np.nan_to_num(self.CCC_loss(arousal_preds[i], arousal_labels[i])) for i in range(batch_size)]
        valence_losses = [np.nan_to_num(self.CCC_loss(valence_preds[i], valence_labels[i])) for i in range(batch_size)]
        # print(arousal_losses)
        # 计算所有序列的平均 CCC 损失
        average_arousal_loss = np.mean(arousal_losses)
        # print(average_arousal_loss)
        average_valence_loss = np.mean(valence_losses)


        val_ccc = self.CCC_total(average_arousal_loss, average_valence_loss)

        results = {
            'valpreds': val_preds,
            'vallabels': val_labels,
            'valccc': val_ccc,
            'ccc_A': average_arousal_loss,
            'ccc_V': average_valence_loss
        }

        outputs = f'CCC: {val_ccc:.4f}_Arousal:{average_arousal_loss:.4f}_Valence:{average_valence_loss:.4f}'
        # print(outputs)

        return results, outputs


