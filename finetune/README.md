## Running Instructions

1. Prepare a python environment, e.g.:

```shell script
$ conda create -n spark python=3.8 -y
$ conda activate spark
```

2. Install `PyTorch` and `timm` (better to use `torch~=1.10`, `torchvision~=0.11`, and `timm==0.5.4`) then other python packages:

```shell script
$ pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install tensorboard
$ pip install -r requirements.txt
```

3. Download the pretrained and baseline model from this [link](https://drive.google.com/drive/folders/1MjumQDNd3HwciWDj8OLQIcusyN3xTRiJ?usp=sharing)
   and save the them to the directory


4. Start fine-tuning on Oxford-IIIT Pet Dataset (dataset will be automatically downloaded if not exist):

```shell script
$ cd PATH/TO/Applied-Deep-Learning-CW3/finetune
$ python finetune.py
```
The fine-tuned pretrained and baseline model can be downloaded from this [link](https://drive.google.com/drive/folders/1Nwg05CYvzPM2awR39qaP733kSGx6PcJj?usp=drive_link)

5. The experiment on changing the training set size of the finetune process is performed by [exp_size.ipynb](https://github.com/Christol-Jalen/Applied-Deep-Learning-CW3/blob/main/finetune/exp_size.ipynb).

   Running the code blocks in sequence could generate all of the results of the second experiment of MRP including plottings, segmentation results, and data, which will all be stored in an automatically created folder ```finetune/exp_img```.

