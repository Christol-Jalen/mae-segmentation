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
import math

######################### Class and Functions ###################################
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def dice_score(self, ps, ts):
        """
        Compute the Dice score, a measure of overlap between two sets.
        """
        numerator = torch.sum(ts * ps, dim=(1, 2, 3)) * 2 + self.eps
        denominator = torch.sum(ts, dim=(1, 2, 3)) + torch.sum(ps, dim=(1, 2, 3)) + self.eps
        return numerator / denominator

    def dice_loss(self, ps, ts):
        """
        Compute the Dice loss, which is -1 times the Dice score.
        """
        return -self.dice_score(ps, ts) 

    def dice_binary(self, ps, ts):
        """
        Threshold predictions and true values at 0.5, convert to float, and compute the Dice score.
        """
        ps = (ps >= 0.5).float()
        ts = (ts >= 0.5).float()
        return self.dice_score(ps, ts)
    
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
        sparse_encoder=enc, 
        dense_decoder=LightDecoder(enc.downsample_raito, sbn=False),
        mask_ratio=config['mask_ratio'], 
        densify_norm=config['densify_norm_str'], 
        sbn=config['sbn'],).to(DEVICE)
    spark.eval(), [p.requires_grad_(False) for p in spark.parameters()]

    # Adjusting loading to handle incompatible keys
    pretrained_dict = pretrained_state
    model_dict = spark.state_dict()

    # 1. Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # 2. Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    
    # load the checkpoint
    missing, unexpected = spark.load_state_dict(model_dict, strict=False)
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
            _, _, outputs = model(images, active_b1ff=active_b1ff, vis=True)
            outputs = outputs.sigmoid()  # Ensure outputs are probabilities
            loss = criterion.dice_loss(outputs, masks)
            loss = loss.mean()
            val_loss += loss.item()

            batch_size = images.size(0)
            images_processed += batch_size

    avg_val_loss = 20 * val_loss / images_processed 
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss

def visualize_segmentation(model, loader, device):
    model.eval()
    images, masks = next(iter(loader))  # Get a batch from the loader
    images, _ = pre_process(images, masks)
    images = images.permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        _, _, preds = model(images, active_b1ff=active_b1ff, vis=True)
    preds = preds.sigmoid().cpu()

    # Plotting
    visualize_images_and_masks(images, preds, masks)

# def visualize_images_and_masks(images, masks, num_images=4):
#     """
#     Visualize the first `num_images` images and masks in a batch.

#     Parameters:
#     - images (torch.Tensor): Tensor containing images.
#     - masks (torch.Tensor): Tensor containing corresponding masks.
#     - num_images (int): Number of images and masks to display.
#     """
#     images = images.permute(0, 2, 3, 1)  # Change from BxCxHxW to BxHxWxC for visualization
#     fig, axs = plt.subplots(2, num_images, figsize=(num_images * 4, 8))  # Set up the subplot grid

#     for i in range(num_images):
#         img = images[i].cpu().detach().numpy()
#         if img.min() < 0 or img.max() > 1:
#             # Normalize to [0, 1]
#             img = (img - img.min()) / (img.max() - img.min())

#         # Display image
#         ax = axs[0, i]
#         ax.imshow(img, interpolation='nearest')
#         ax.axis('off')
#         ax.set_title('Image')

#         # Display mask
#         ax = axs[1, i]
#         mask = masks[i].squeeze()  # Remove channel dim if it's there
#         ax.imshow(mask.cpu().detach().numpy(), cmap='gray', interpolation='nearest')
#         ax.axis('off')
#         ax.set_title('Mask')

#     plt.tight_layout()
#     plt.show()
    
def visualize_images_and_masks(images, outputs, masks, num_images=1):
    """
    Visualize the first `num_images` images and masks in a batch.

    Parameters:
    - images (torch.Tensor): Tensor containing images.
    - masks (torch.Tensor): Tensor containing corresponding masks.
    - num_images (int): Number of images and masks to display.
    """
    images = images.permute(0, 2, 3, 1)  # Change from BxCxHxW to BxHxWxC for visualization
    fig, axs = plt.subplots(3, num_images, figsize=(num_images * 4, 8))  # Set up the subplot grid

    for i in range(num_images):
        img = images[i].cpu().detach().numpy()
        if img.min() < 0 or img.max() > 1:
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min())

    # Display image
    ax = axs[0]
    ax.imshow(img, interpolation='nearest')
    ax.axis('off')
    ax.set_title('Image')
    ax.set_title('Input')
    
    # Display output
    ax = axs[1]
    out = outputs.squeeze()  # Remove channel dim if it's there
    ax.imshow(out.cpu().detach().numpy(), interpolation='nearest')
    ax.axis('off')
    ax.set_title('Prediction')

    # Display mask
    ax = axs[2]
    mask = masks.squeeze()  # Remove channel dim if it's there
    ax.imshow(mask.cpu().detach().numpy(), cmap='gray', interpolation='nearest')
    ax.axis('off')
    ax.set_title('Ground Truth')

    plt.tight_layout()
    plt.show()


def post_process(output):
    """
    Post-processes the output segmentation mask tensor.
    
    Args:
        output (torch.Tensor): Output segmentation mask tensor of shape [1, 1, 224, 224].
        
    Returns:
        torch.Tensor: Post-processed segmentation mask tensor.
    """
    threshold = 0.5
    device = output.device  # Get the device of the output tensor
    processed_output = torch.where(output > threshold, torch.tensor(1, device=device), torch.tensor(0, device=device))
    return processed_output

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
minibatch_size = 1
network_size = 16
learning_rate = 1e-4
num_epochs = 500
freq_info = 1
freq_save = 100
save_path = "results_pt"
best_val_loss = float('inf')

active_b1ff = torch.tensor([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
], device=DEVICE).bool().reshape(1, 1, 7, 7)

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
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Training loop
print("start training")
for epoch in range(200): 
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
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
        _, _, outputs = model(images, active_b1ff=active_b1ff, vis=True)  

        outputs = torch.sigmoid(outputs) 
        loss = criterion.dice_loss(outputs, masks)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        batch_size = images.size(0)
        images_processed += batch_size  # Update the counter by the number of images in the current batch

        # Report the current average loss after every 500 images
        if images_processed % 1000 == 0:
            print(f"Processed {images_processed} images, Current Loss: {running_loss/images_processed:.4f}")
            # visualize_images_and_masks(outputs, masks)
            # Print shapes of images, masks, and outputs
            # print("Shapes - Images:", images.shape, "Masks:", masks.shape, "Outputs:", outputs.shape)

            # # Print 4 pixel values for each of images, masks, and outputs
            # print("Pixel Values - Images:") 
            # print(images[:,:,:4,:4])
            # print("Masks:")
            # print(masks[:,:,:4,:4])
            # print("Outputs:")
            # print(outputs[:,:,:4,:4])

            # outputs  = post_process(outputs)
            # visualize_images_and_masks(images, outputs, masks)

    print(f"Epoch {epoch+1}, Loss: {running_loss/images_processed}")
    val_loss = validate(model, loader_val, criterion, DEVICE)
    

# save the model
torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
print("Model saved.")

# After training, visualize segmentation output
visualize_segmentation(model, loader_val, DEVICE)
