from .iemocap import IEMOCAP
from .cmudata import CMUDATA
from .mer2023 import MER2023
from .sims import SIMS
from .meld import MELD
from .simsv2 import SIMSv2
from .crossdim import CROSSDIM
from .crossdis import CROSSDIS
from .odyssey import ODYSSEY
from .abaw import ABAW

DIM_DATASET = ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2','ABAW']
DIS_DATASET = ['IEMOCAPFour', 'IEMOCAPSix', 'MER2023', 'MELD', 'ODYSSEY']

# 输入数据库名称，得到 dataloaders
class get_dataloaders:

    def __init__(self, args):

        if args.train_dataset is None:
            DATALOADER_MAP = {
                
                'IEMOCAPFour': IEMOCAP,
                'IEMOCAPSix':  IEMOCAP,
                'CMUMOSI':     CMUDATA,
                'CMUMOSEI':    CMUDATA,
                'MER2023':     MER2023,
                'SIMS': SIMS,
                'SIMSv2': SIMSv2,
                'MELD': MELD,
                'ODYSSEY': ODYSSEY,
                'ABAW': ABAW,
            }
            self.dataloader = DATALOADER_MAP[args.dataset](args)
        elif args.train_dataset in DIM_DATASET:
            assert args.test_dataset in DIM_DATASET
            self.dataloader = CROSSDIM(args)
        elif args.train_dataset in DIS_DATASET:
            assert args.test_dataset in DIS_DATASET
            self.dataloader = CROSSDIS(args)

    def get_loaders(self):
        return self.dataloader.get_loaders()
    
    def calculate_results(self, emo_probs=[], emo_labels=[], val_preds=[], val_labels=[]):
        return self.dataloader.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
    
