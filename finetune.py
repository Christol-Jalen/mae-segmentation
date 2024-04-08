import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from loader import H5ImageLoader
import os
from decoder import LightDecoder
from encoder import SparseEncoder
from models import build_sparse_encoder
from spark import SparK
# import timm

# specify the device to use
USING_GPU_IF_AVAILABLE = True

_ = torch.empty(1)
if torch.cuda.is_available() and USING_GPU_IF_AVAILABLE:
    _ = _.cuda()
DEVICE = _.device
print(f'[DEVICE={DEVICE}]')


os.environ["CUDA_VISIBLE_DEVICES"]="0"
DATA_PATH = './data'


## settingsbatch_idxbatch_idx
minibatch_size = 4
network_size = 16
learning_rate = 1e-4
num_epochs = 500
freq_info = 1
freq_save = 100
save_path = "results_pt"

if not os.path.exists(save_path):
    os.makedirs(save_path)


## data loader
loader_train = H5ImageLoader(DATA_PATH+'/images_train.h5', minibatch_size, DATA_PATH+'/labels_train.h5')
loader_val = H5ImageLoader(DATA_PATH+'/images_val.h5', 20, DATA_PATH+'/labels_val.h5')

############################################################
def build_spark(your_own_pretrained_ckpt: str = ''):
    if len(your_own_pretrained_ckpt) > 0 and os.path.exists(your_own_pretrained_ckpt):
        all_state = torch.load(your_own_pretrained_ckpt, map_location='cpu')
        input_size, model_name = all_state['input_size'], all_state['arch']
        pretrained_state = all_state['module']
        print(f"[in function `build_spark`] your ckpt `{your_own_pretrained_ckpt}` loaded;  don't forget to modify IMAGENET_RGB_MEAN and IMAGENET_RGB_MEAN above if needed")
    else:
        # download and load the checkpoint
        input_size, model_name, file_path, ckpt_link = {
            'ResNet50': (224, 'resnet50', 'res50_withdecoder_1kpretrained_spark_style.pth', 'https://drive.google.com/file/d/1STt3w3e5q9eCPZa8VzcJj1zG6p3jLeSF/view?usp=share_link'),
            'ResNet101': (224, 'resnet101', 'res101_withdecoder_1kpretrained_spark_style.pth', 'https://drive.google.com/file/d/1GjN48LKtlop2YQre6---7ViCWO-3C0yr/view?usp=share_link'),
            'ResNet152': (224, 'resnet152', 'res152_withdecoder_1kpretrained_spark_style.pth', 'https://drive.google.com/file/d/1U3Cd94j4ZHfYR2dUjWmsEWfjP6Opx4oo/view?usp=share_link'),
            'ResNet200': (224, 'resnet200', 'res200_withdecoder_1kpretrained_spark_style.pth', 'https://drive.google.com/file/d/13AFSqvIr0v-2hmb4DzVza45t_lhf2CnD/view?usp=share_link'),
            'ConvNeXt-S': (224, 'convnext_small', 'cnxS224_withdecoder_1kpretrained_spark_style.pth', 'https://drive.google.com/file/d/1bKvrE4sNq1PfzhWlQJXEPrl2kHqHRZM-/view?usp=share_link'),
            'ConvNeXt-L': (384, 'convnext_large', 'cnxL384_withdecoder_1kpretrained_spark_style.pth', 'https://drive.google.com/file/d/1ZI9Jgtb3fKWE_vDFEly29w-1FWZSNwa0/view?usp=share_link')
        }['ResNet50']  # you can choose any model here
        assert os.path.exists(file_path), f'please download checkpoint {file_path} from {ckpt_link}'
        pretrained_state = torch.load(file_path, map_location='cpu')
        if 'module' in pretrained_state:
            pretrained_state = pretrained_state['module']

    # build a SparK model
    config = pretrained_state['config']
    enc: SparseEncoder = build_sparse_encoder(model_name, input_size=input_size)
    spark = SparK(
        sparse_encoder=enc, dense_decoder=LightDecoder(enc.downsample_raito, sbn=False),
        mask_ratio=config['mask_ratio'], densify_norm=config['densify_norm_str'], sbn=config['sbn'],
    ).to(DEVICE)
    spark.eval(), [p.requires_grad_(False) for p in spark.parameters()]
    
    # load the checkpoint
    missing, unexpected = spark.load_state_dict(pretrained_state, strict=False)
    assert len(missing) == 0, f'load_state_dict missing keys: {missing}'
    assert len(unexpected) == 0, f'load_state_dict unexpected keys: {unexpected}'
    del pretrained_state
    return spark, input_size

def pre_process(images, labels):
    # Convert each numpy array in `images` to a PyTorch tensor and stack
    images = torch.stack([torch.tensor(img).float() for img in images])
    # Similarly, ensure labels are tensors, then stack and add an extra dimension
    labels = torch.stack([torch.tensor(lbl).unsqueeze(-1).float() for lbl in labels])
    return images, labels

###########################################################
model, input_size = build_spark()
print("model built")

# # Load the pretrained model
# model = timm.create_model('resnet50', pretrained=False, num_classes=2)  # pet + background
# model.load_state_dict(torch.load("resnet50_1kpretrained_timm_style.pth"), strict=False)

# Fine tune the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
print("start training")
for epoch in range(10):  # Number of epochs
    model.train()
    running_loss = 0.0
    for images, masks in loader_train:  #  masks are directly the second element
        images, masks = pre_process(images, masks)
        images = images.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        print("Shape of images tensor:", images.shape)
        print("Shape of masks tensor:", masks.shape)
        images, masks = images.to(DEVICE), masks.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images) 

        # Ensure outputs and masks are correctly aligned; might need to adjust dimensions
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader_train)}")