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

3. Download pretrained model from this [link](https://drive.google.com/file/d/1STt3w3e5q9eCPZa8VzcJj1zG6p3jLeSF/view?usp=share_link
   ) save the pretrained model to the current directory.

   0 epoch: [link]https://drive.google.com/drive/folders/14XH8F7En_C-8zGiy2zob3VR-WLVUT1sy
   
   90 epoch: [link]https://drive.google.com/file/d/1Ry_hsxv9EeqYPi43ODqEAVn4PjXLpGHP/view?usp=drive_link

4. Start fine-tuning on Oxford-IIIT Pet Dataset (dataset will be automatically downloaded if not exist):

```shell script
$ python finetune.py
```

