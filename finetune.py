import torch
import torch.optim as optim
from loader import H5ImageLoader
import os
from decoder import LightDecoder
from encoder import SparseEncoder
from models import build_sparse_encoder
from spark import SparK
from loss import DiceLoss
import torch.distributed as dist
import matplotlib.pyplot as plt


# Set the environment variables for distributed training
os.environ['MASTER_ADDR'] = 'localhost'  # or another appropriate address
os.environ['MASTER_PORT'] = '12355'  # choose an open port

# Initialise distribute for single GPU training
dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)

# Specify the device to use
USING_GPU_IF_AVAILABLE = True

ir_ = torch.empty(1)
if torch.cuda.is_available() and USING_GPU_IF_AVAILABLE:
    ir_ = ir_.cuda()
DEVICE = ir_.device
print(f'[DEVICE={DEVICE}]')


os.environ["CUDA_VISIBLE_DEVICES"]="0"
DATA_PATH = './data'


## Training parameters
minibatch_size = 1
learning_rate = 1e-4
num_epochs = 200
criterion = DiceLoss()
save_path = "results_pt"

if not os.path.exists(save_path):
    os.makedirs(save_path)


## Data loader
loader_train = H5ImageLoader(DATA_PATH+'/images_train.h5', minibatch_size, DATA_PATH+'/labels_train.h5')
loader_val = H5ImageLoader(DATA_PATH+'/images_val.h5', 20, DATA_PATH+'/labels_val.h5')

def main():

    model = build_spark('res50_withdecoder_1kpretrained_spark_style.pth')
    print("model built")

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("start training")
    for epoch in range(num_epochs): 
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
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images, active_b1ff=None)  

            outputs = torch.sigmoid(outputs) 
            loss = criterion.dice_loss(outputs, masks)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            batch_size = images.size(0)
            images_processed += batch_size

            # Report the current average loss after every 500 images
            if images_processed % 300 == 0:
                print(f"Processed {images_processed} images, Current Loss: {running_loss/images_processed:.4f}")
                visualize_images_outputs_and_masks(images, outputs, masks)
                

        print(f"Epoch {epoch+1}, Loss: {running_loss/images_processed}")
        val_loss = validate(model, loader_val, criterion, DEVICE)
        

    # Save the model
    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
    print("Model saved.")

    # After training, visualize segmentation output
    visualize_validation(model, loader_val, DEVICE)


def build_spark(your_own_pretrained_ckpt: str):
        
    input_size, model_name = 224, 'resnet50'
    pretrained_state = torch.load(your_own_pretrained_ckpt, map_location='cpu')
    print(f"[in function `build_spark`] your ckpt `{your_own_pretrained_ckpt}` loaded")

    # Build a SparK model
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

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    
    # Load the checkpoint
    missing, unexpected = spark.load_state_dict(model_dict, strict=False)
    assert len(missing) == 0, f'load_state_dict missing keys: {missing}'
    assert len(unexpected) == 0, f'load_state_dict unexpected keys: {unexpected}'
    del pretrained_state
    return spark

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
            outputs = model(images, active_b1ff=None)
            outputs = outputs.sigmoid() 
            loss = criterion.dice_loss(outputs, masks)
            loss = loss.mean()
            val_loss += loss.item()
            batch_size = images.size(0)
            images_processed += batch_size

    avg_val_loss = 20 * val_loss / images_processed 
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss

def visualize_validation(model, loader, device):
    model.eval()
    images, masks = next(iter(loader))  # Get a batch from the loader
    images, _ = pre_process(images, masks)
    images = images.permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        preds = model(images, active_b1ff=None)
    preds = preds.sigmoid().cpu()

    # Plotting
    visualize_images_outputs_and_masks(images, preds, masks)

    
def visualize_images_outputs_and_masks(images, outputs, masks, num_images=1):
  
    images = images.permute(0, 2, 3, 1)  # Change from BxCxHxW to BxHxWxC for visualization
    _, axs = plt.subplots(3, num_images, figsize=(num_images * 4, 8))  # Set up the subplot grid

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
        outputs  = post_process(outputs)
        ax = axs[1]
        out = outputs.squeeze()  # Remove channel dim if it's there
        ax.imshow(out.cpu().detach().numpy(), interpolation='nearest')
        ax.axis('off')
        ax.set_title('Prediction')

        # Display mask
        ax = axs[2]
        mask = masks.squeeze()  # Remove channel dim if it's there
        ax.imshow(mask.cpu().detach().numpy(), interpolation='nearest')
        ax.axis('off')
        ax.set_title('Ground Truth')

        plt.tight_layout()
        plt.show()


def post_process(output):
  
    threshold = 0.5
    device = output.device  # Get the device of the output tensor
    processed_output = torch.where(output > threshold, torch.tensor(1, device=device), torch.tensor(0, device=device))
    return processed_output



if __name__ == '__main__':
    main()
