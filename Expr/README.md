
#  Affwild2-ABAW3 @ CVPR 2022
## Task: EXPRESSION CLASSIFICATION

Name: Kim Ngan Ngan, Hong-Hai Nguyen, Van-Thong Huynh, Soo-Hyung Kim

**Paper: Facial Expression Classification using Fusion of Deep Neural Network in Video** [here](https://openaccess.thecvf.com/content/CVPR2022W/ABAW/papers/Phan_Facial_Expression_Classification_Using_Fusion_of_Deep_Neural_Network_in_CVPRW_2022_paper.pdf)

$5^{th}$ place at [Affwild2-ABAW3 @ CVPR 2022](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/)


### Set up environment
+ Create a python new_env environment using conda or other tools.

+ Instead packages in requirements.txt
```bash
pip install -r requirements.txt
```
+ Activate new_env environment
```bash
conda activate new_env
```
### How to train?
+  Create dataset-folder that contains **cropped_aligned** folder and **3rd_ABAW_Annotations folder**

+  Run **data_preparation.py** in **tools** to create .npy file in out-data-folder
```bash
python data_preparation.py --root_dir path/to/dataset-folder --out_dir path/to/out-data-folder
```
+  Edit .yaml file in **conf** with

    OUT_DIR: path to save tmp file that contains .ckpt and .yaml

    DATA_DIR: path to .npy file in out-data-folder

    MODEL_NAME: choose one in {combine, no_att_trans, only_att, only_trans} types
```bash
python main.py --cfg ./conf/EXPR_baseline.yaml
```

### How to predict?

+  Create batch-1-2-folder that contains entire videos in batch_1 and batch_2 folder

+  Create testset-folder that contains **EXPR_test_set_release.txt** file

+  Run prepare_test_data.py in **tools** to create EXPR_test.npy in out-data-folder
```bash
python prepare_test_data.py --root_video_dir path/to/batch-1-2-folder --dataset_dir path/to/out-data-folder
```
+  Get prediction file of test set at OUT_DIR
```bash
python main.py --cfg /path/to/config-yaml-file
```

