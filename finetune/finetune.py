import torch
import torch.optim as optim
from loader import H5ImageLoader
import os
from decoder import LightDecoder
from encoder import SparseEncoder
from network import build_sparse_encoder
from spark import SparK
from loss import DiceLoss
import torch.distributed as dist
# import matplotlib.pyplot as plt
import data
from PIL import Image
import numpy as np

# Prepare the dataset
# ratio_train can be chosen in [0, 0.85], as test set ratio is fixed at 10%
# split_data should be set to True if: 1. the dataset haven't been downloaded before or 2. the ratio_train is changed
data.prepare_dataset(ratio_train = 0.7, split_data = False)

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
minibatch_size = 4
learning_rate = 1e-4
num_epochs = 50
criterion = DiceLoss()
model_path = "models"

# if not os.path.exists(save_path):
#     os.makedirs(save_path)


## Data loader
loader_train = H5ImageLoader(DATA_PATH+'/images_train.h5', minibatch_size, DATA_PATH+'/labels_train.h5')
loader_val = H5ImageLoader(DATA_PATH+'/images_val.h5', 20, DATA_PATH+'/labels_val.h5')
loader_test = H5ImageLoader(DATA_PATH+'/images_test.h5', 20, DATA_PATH+'/labels_test.h5')
print("Dataset Loaded: num_train: %d, num_val: %d, num_test: %d" % (loader_train.num_images, loader_val.num_images, loader_test.num_images))

def main():

    # Build the model
    model = build_spark('resnet50_90epochs_imagenet100pretrained.pth') # Change this to the path of your own pretrained model
    print("model built")

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Finetuning loop
    print("start finetuning")
    for epoch in range(num_epochs): 
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)
        running_loss = 0.0
        batches_processed = 0
        for images, masks in loader_train:

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
            batches_processed += 1

            # Report the current average loss after every 500 batches
            # if batches_processed % 500 == 0:
            #      print(f"Processed {batches_processed} batches, Current Loss: {running_loss/batches_processed:.4f}")
            #      val_loss, val_accuracy = validate_and_test(model, loader_val, criterion, DEVICE, vis=True)
            #      print(f"Current Validation Loss: {val_loss}")
            #      print(f"Current Validation Accuracy: {val_accuracy}%")
                
        print(f"Epoch {epoch+1}, Loss: {running_loss/batches_processed}")
        val_loss, val_accuracy = validate_and_test(model, loader_val, criterion, DEVICE, vis=False)
        # print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy}%")
        # visualize_images_outputs_and_masks(images, outputs, masks)

    # Test the model after training
    print("Training finished. Testing the model.")
    test_loss, test_accuracy = validate_and_test(model, loader_test, criterion, DEVICE, vis=True)
    print(f"Training finished.\nTest Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")  

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))
    print("Model saved.")



def build_spark(your_own_pretrained_ckpt: str):
        
    input_size, model_name = 224, 'resnet50'
    pretrained_state = torch.load(your_own_pretrained_ckpt, map_location='cpu')
    print(f"[in function `build_spark`] your ckpt `{your_own_pretrained_ckpt}` loaded")

    # Build a SparK model
    #print(pretrained_state.keys())
    enc: SparseEncoder = build_sparse_encoder(model_name, input_size=input_size)
    spark = SparK(
        sparse_encoder=enc, 
        dense_decoder=LightDecoder(enc.downsample_raito, sbn=False)
        ).to(DEVICE)
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

def validate_and_test(model, loader, criterion, device, vis):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    total_batches = 0
    total_correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():  # No gradients needed
        for images, masks in loader:
            images, masks = pre_process(images, masks)
            images = images.permute(0, 3, 1, 2).to(device)
            masks = masks.permute(0, 3, 1, 2).to(device)
            outputs = model(images, active_b1ff=None)
            outputs = outputs.sigmoid() 
            
            # Calculate loss
            loss = criterion.dice_loss(outputs, masks)
            loss = loss.mean()
            val_loss += loss.item()
            
            # Calculate accuracy
            predicted_masks = (outputs > 0.5).float()  # threshold of 0.5 for binarization
            correct_pixels = torch.sum(predicted_masks == masks).item()
            total_correct_pixels += correct_pixels
            total_pixels += torch.numel(masks)
            
            total_batches += 1
    
    avg_loss = val_loss / total_batches 
    accuracy = total_correct_pixels / total_pixels * 100
    model.train() # Put the model back to train mode

    if vis:
        visualize_images_outputs_and_masks(images, outputs, masks)
    
    return avg_loss, accuracy

    
def concatenate_images(image_list):
    widths, heights = zip(*(i.size for i in image_list))
    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in image_list:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    return new_im

def visualize_images_outputs_and_masks(images, outputs, masks, num_images=minibatch_size):
    # Post-process outputs
    outputs = post_process(outputs)
    
    # Ensure images, outputs, and masks are tensor type and check dimensions
    if not (images.ndim == 4 and outputs.ndim in [3, 4] and masks.ndim in [3, 4]):
        raise ValueError("Invalid input dimensions")

    # Adjust data for PIL
    images = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    outputs = outputs.squeeze(1).cpu().detach().numpy() if outputs.ndim == 4 else outputs.cpu().detach().numpy()
    masks = masks.squeeze(1).cpu().detach().numpy() if masks.ndim == 4 else masks.cpu().detach().numpy()

    # Normalize and convert to uint8
    for i in range(num_images):
        img = ((images[i] - images[i].min()) / (images[i].max() - images[i].min()) * 255).astype(np.uint8)
        out = (outputs[i] * 255).astype(np.uint8)
        msk = (masks[i] * 255).astype(np.uint8)

        pil_img = Image.fromarray(img, 'RGB')
        pil_out = Image.fromarray(out, 'L').convert('RGB')
        pil_msk = Image.fromarray(msk, 'L').convert('RGB')

        # Combine images vertically
        combined_image = concatenate_images([pil_img, pil_out, pil_msk])
        # Save the image
        combined_image.save(f"result_{i}.png")



def post_process(output):
  
    threshold = 0.5
    device = output.device  # Get the device of the output tensor
    processed_output = torch.where(output > threshold, torch.tensor(1, device=device), torch.tensor(0, device=device))
    return processed_output



if __name__ == '__main__':
    main()
