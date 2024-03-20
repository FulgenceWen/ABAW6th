import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *
from toolkit.data import get_datasets


class ODYSSEY:

    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.num_folder = 5
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.label_path = config.PATH_TO_LABEL[args.dataset]

        self.dataset = args.dataset
        assert self.dataset in ['ODYSSEY']

        # update args
        args.output_dim1 = 0
        args.output_dim2 = 1
        args.metric_name = 'Act'

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
            else:
                dataloader = DataLoader(dataset,
                                        batch_size=self.batch_size,
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
        return names, labels

    def CCC_loss(self, pred, lab, is_numpy=False):
        if is_numpy:
            pred = torch.Tensor(pred).float().cuda()
            lab = torch.Tensor(lab).float().cuda()

        m_pred = torch.mean(pred)
        m_lab = torch.mean(lab)

        sum_pred_lab = torch.sum((pred - m_pred) * (lab - m_lab))
        sum_pred_sq = torch.sum((pred - m_pred) ** 2)
        sum_lab_sq = torch.sum((lab - m_lab) ** 2)

        corr = sum_pred_lab / (torch.sqrt(sum_pred_sq) * torch.sqrt(sum_lab_sq))

        ccc = (2 * corr) / (torch.var(pred, unbiased=False) + torch.var(lab, unbiased=False) +
                            (m_pred - m_lab) ** 2)
        return ccc

    def calculate_ccc(self, emo_probs=[], emo_labels=[]):

        # 将emo_probs和emo_labels转换为tensor，并在有可用的情况下移动到cuda
        total_pred = torch.tensor(emo_probs, dtype=torch.float32).view(-1, 1)
        total_y = torch.tensor(emo_labels, dtype=torch.float32).view(-1, 1)

        if torch.cuda.is_available():
            total_pred = total_pred.cuda()
            total_y = total_y.cuda()

        ccc = self.CCC_loss(total_pred, total_y, is_numpy=False)

        ccc_value = ccc.item()

        results = {
            'val_preds': emo_probs,
            'val_labels': emo_labels,
            'ccc': ccc_value,
        }

        outputs = f'CCC: {ccc_value:.4f}'

        return results, outputs







