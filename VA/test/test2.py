import os

import numpy as np
path='/data/wenzhuofan/Data/ABAW/Feature/features/wavlm-large-FRA'
for file in sorted(os.listdir(path)):
    if file.endswith('.npy'):
        data=np.load(os.path.join(path,file))
        print(file)
        print(data.shape)