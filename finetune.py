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
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np

######################### Class and Functions ###################################
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.sigmoid()  # Applying sigmoid to squash outputs to [0,1] range
        intersection = (y_pred * y_true).sum(dim=[2,3])
        dice_score = (2. * intersection + self.smooth) / (y_pred.sum(dim=[2,3]) + y_true.sum(dim=[2,3]) + self.smooth)
        return 1 - dice_score.mean()
    
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

def validate(model, loader_val, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    images_processed = 0
    with torch.no_grad():  # No gradients needed
        for images, masks in loader_val:
            images, masks = pre_process(images, masks)
            images = images.permute(0, 3, 1, 2).to(device)
            masks = masks.permute(0, 3, 1, 2).to(device)
            _, outputs, _ = model(images, active_b1ff=None, vis=True)
            outputs = outputs.sigmoid()
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            batch_size = images.size(0)
            images_processed += batch_size

    avg_val_loss = val_loss / images_processed
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss

def visualize_segmentation(model, loader, device):
    model.eval()
    images, masks = next(iter(loader))  # Get a batch from the loader
    images, _ = pre_process(images, masks)
    images = images.permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        _, preds, _ = model(images, active_b1ff=None, vis=True)
    preds = preds.sigmoid().cpu()

    # Plotting
    plt.figure(figsize=(10, 4))
    for i in range(min(4, images.size(0))):  # Show 4 images 
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0).numpy().astype(np.uint8))
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(2, 4, i + 5)
        plt.imshow(preds[i].squeeze().numpy(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    plt.show()

###########################################################

# Set the environment variables for distributed training
os.environ['MASTER_ADDR'] = 'localhost'  # or another appropriate address
os.environ['MASTER_PORT'] = '12355'  # choose an open port

# Initialise distribute for single GPU training
dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)

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
best_val_loss = float('inf')

if not os.path.exists(save_path):
    os.makedirs(save_path)


## data loader
loader_train = H5ImageLoader(DATA_PATH+'/images_train.h5', minibatch_size, DATA_PATH+'/labels_train.h5')
loader_val = H5ImageLoader(DATA_PATH+'/images_val.h5', 20, DATA_PATH+'/labels_val.h5')


model, input_size = build_spark()
print("model built")

# # Load the pretrained model
# model = timm.create_model('resnet50', pretrained=False, num_classes=2)  # pet + background
# model.load_state_dict(torch.load("resnet50_1kpretrained_timm_style.pth"), strict=False)

# Fine tune the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
# criterion = nn.CrossEntropyLoss()
criterion = DiceLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
print("start training")
for epoch in range(3): 
    model.train()
    running_loss = 0.0
    images_processed = 0
    for i, (images, masks) in enumerate(loader_train):
        # Pre-process inputs and masks
        images, masks = pre_process(images, masks)
        images = images.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        # For binary segmentation, masks might need to be squeezed to remove the channel dimension if it's 1
        # masks = masks.squeeze(1)  
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        _, outputs, _ = model(images, active_b1ff=None, vis=True)
        outputs = outputs.sigmoid()  # Apply sigmoid to outputs to squash them to [0,1] range

        outputs.requires_grad_()
        masks.requires_grad_()
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        batch_size = images.size(0)
        images_processed += batch_size  # Update the counter by the number of images in the current batch

        # Report the current average loss after every 500 images
        if images_processed % 500 == 0:
            print(f"Processed {images_processed} images, Current Loss: {running_loss/images_processed:.4f}")

    print(f"Epoch {epoch+1}, Loss: {running_loss/images_processed}")
    val_loss = validate(model, loader_val, criterion, DEVICE)

# save the model
torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
print("Model saved.")

# After training, visualize segmentation output
visualize_segmentation(model, loader_val, DEVICE)
