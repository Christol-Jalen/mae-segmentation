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
import data
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Prepare the dataset
# ratio_train can be chosen in [0, 0.85], as test set ratio is fixed at 10%
# split_data should be set to True if: 1. the dataset haven't been downloaded before or 2. the ratio_train is changed
data.prepare_dataset(ratio_train = 0.7, split_data = True)

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
minibatch_size = 32
learning_rate = 1e-4
num_epochs = 50
criterion = DiceLoss()
model_path = "models"
results_path = "results"

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(results_path):
    os.makedirs(results_path)

if minibatch_size<20:
    ## Data loader
    loader_train = H5ImageLoader(DATA_PATH+'/images_train.h5', minibatch_size, DATA_PATH+'/labels_train.h5')
    loader_val = H5ImageLoader(DATA_PATH+'/images_val.h5', 20, DATA_PATH+'/labels_val.h5')
    loader_test = H5ImageLoader(DATA_PATH+'/images_test.h5', 20, DATA_PATH+'/labels_test.h5')
else:
    ## Data loader
    loader_train = H5ImageLoader(DATA_PATH+'/images_train.h5', minibatch_size, DATA_PATH+'/labels_train.h5')
    loader_val = H5ImageLoader(DATA_PATH+'/images_val.h5', minibatch_size, DATA_PATH+'/labels_val.h5')
    loader_test = H5ImageLoader(DATA_PATH+'/images_test.h5', minibatch_size, DATA_PATH+'/labels_test.h5')

print("Dataset Loaded: num_train: %d, num_val: %d, num_test: %d" % (loader_train.num_images, loader_val.num_images, loader_test.num_images))

def main():

    # Build the model
    model = build_spark('model_200epochs.pth') # Change this to the path of your own pretrained model
    print("model built")

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare lists to store loss and accuracy histories
    train_losses = []
    val_losses = [] 
    train_accuracies = []
    val_accuracies = []

    # Finetuning loop 
    print("start finetuning")
    for epoch in range(num_epochs):
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)
        running_loss = 0.0
        batches_processed = 0
        total_correct_pixels = 0
        total_pixels = 0
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

            # Calculate accuracy
            predicted_masks = (outputs > 0.5).float()  # threshold of 0.5 for binarization
            correct_pixels = torch.sum(predicted_masks == masks).item()
            total_correct_pixels += correct_pixels
            total_pixels += torch.numel(masks)

        train_loss = running_loss / batches_processed
        train_accuracy = total_correct_pixels / total_pixels * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")
        print(f"Epoch {epoch+1}, Training Accuracy: {train_accuracy:.2f}%")

        # Validate the model
        val_loss, val_accuracy = validate_and_test(model, loader_val, criterion, DEVICE, vis=False)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.2f}%")

    # Plot and save the loss and accuracy curves using Pillow
    plot_loss_and_accuracy_pillow(train_losses, val_losses, train_accuracies, val_accuracies)
    # Test the model after training
    print("Training finished. Testing the model.")
    test_loss, test_accuracy = validate_and_test(model, loader_test, criterion, DEVICE, vis=True)
    print(f"Training finished.\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Save the test results
    with open(os.path.join(results_path, "test_results.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")

    # Save the model 
    torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))
    print("Model saved.")


def build_spark(your_own_pretrained_ckpt: str):

    input_size, model_name = 224, 'resnet50'
    pretrained_state = torch.load(your_own_pretrained_ckpt, map_location='cpu')
    print(f"[in function `build_spark`] your ckpt `{your_own_pretrained_ckpt}` loaded")

    # Build a SparK model
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
        combined_image.save(os.path.join(results_path, f"result_{i}.png"))


def post_process(output):

    threshold = 0.5
    device = output.device  # Get the device of the output tensor
    processed_output = torch.where(output > threshold, torch.tensor(1, device=device), torch.tensor(0, device=device))
    return processed_output


def plot_loss_and_accuracy_pillow(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Define colors
    train_color = (51, 153, 255)  # Blue
    val_color = (255, 51, 51)  # Red
    grid_color = (200, 200, 200)  # Light gray
    text_color = (0, 0, 0)  # Black

    # Define font
    title_font = ImageFont.truetype("utils/Arial.ttf", 24)
    label_font = ImageFont.truetype("utils/Arial.ttf", 14)

    # Calculate the maximum loss and accuracy values
    max_loss = max(max(train_losses), max(val_losses))
    min_loss = min(min(train_losses), min(val_losses))
    max_accuracy = max(max(train_accuracies), max(val_accuracies))
    min_accuracy = min(min(train_accuracies), min(val_accuracies))

    # Define the padding and spacing
    padding = 50
    y_spacing = (height - padding * 2) / 2

    # Plot loss
    loss_y_scale = (y_spacing - padding) / (max_loss - min_loss)
    loss_y_offset = padding + (max_loss - min_loss) * loss_y_scale
    for i in range(len(epochs)):
        x = padding + i * (width - padding * 2) // (len(epochs) - 1)
        y_train = loss_y_offset - (train_losses[i] - min_loss) * loss_y_scale
        y_val = loss_y_offset - (val_losses[i] - min_loss) * loss_y_scale
        draw.ellipse((x - 3, y_train - 3, x + 3, y_train + 3), fill=train_color)
        draw.ellipse((x - 3, y_val - 3, x + 3, y_val + 3), fill=val_color)
        if i > 0:
            prev_x = padding + (i - 1) * (width - padding * 2) // (len(epochs) - 1)
            prev_y_train = loss_y_offset - (train_losses[i - 1] - min_loss) * loss_y_scale
            prev_y_val = loss_y_offset - (val_losses[i - 1] - min_loss) * loss_y_scale
            draw.line((prev_x, prev_y_train, x, y_train), fill=train_color, width=2)
            draw.line((prev_x, prev_y_val, x, y_val), fill=val_color, width=2)

    # Add grid lines and labels for loss
    num_loss_ticks = 5
    for i in range(num_loss_ticks + 1):
        y = loss_y_offset - i * (y_spacing - padding) // num_loss_ticks
        draw.line((padding, y, width - padding, y), fill=grid_color)
        loss_value = min_loss + i * (max_loss - min_loss) / num_loss_ticks
        draw.text((5, y - 6), f"{loss_value:.3f}", font=label_font, fill=text_color)

    draw.text((padding, padding - 30), "Loss", font=label_font, fill=text_color)

    # Plot accuracy
    accuracy_y_scale = (y_spacing - padding) / (max_accuracy - min_accuracy)
    accuracy_y_offset = height - padding  # Set baseline at the bottom edge
    for i in range(len(epochs)):
        x = padding + i * (width - padding * 2) // (len(epochs) - 1)
        y_train = accuracy_y_offset - (train_accuracies[i] - min_accuracy) * accuracy_y_scale
        y_val = accuracy_y_offset - (val_accuracies[i] - min_accuracy) * accuracy_y_scale
        draw.ellipse((x - 3, y_train - 3, x + 3, y_train + 3), fill=train_color)
        draw.ellipse((x - 3, y_val - 3, x + 3, y_val + 3), fill=val_color)
        if i > 0:
            prev_x = padding + (i - 1) * (width - padding * 2) // (len(epochs) - 1)
            prev_y_train = accuracy_y_offset - (train_accuracies[i - 1] - min_accuracy) * accuracy_y_scale
            prev_y_val = accuracy_y_offset - (val_accuracies[i - 1] - min_accuracy) * accuracy_y_scale
            draw.line((prev_x, prev_y_train, x, y_train), fill=train_color, width=2)
            draw.line((prev_x, prev_y_val, x, y_val), fill=val_color, width=2)

    # Add grid lines and labels for accuracy
    num_accuracy_ticks = 5
    for i in range(num_accuracy_ticks + 1):
        y = accuracy_y_offset - i * (y_spacing - padding) // num_accuracy_ticks
        draw.line((padding, y, width - padding, y), fill=grid_color)
        accuracy_value = (min_accuracy + i * (max_accuracy - min_accuracy) / num_accuracy_ticks) / 100
        draw.text((5, y - 6), f"{accuracy_value:.2%}", font=label_font, fill=text_color)

    # Add x-axis labels (epochs)
    num_epoch_ticks = min(len(epochs), 10)  # Limit the number of epoch ticks to 10
    epoch_tick_interval = len(epochs) // num_epoch_ticks
    for i in range(0, len(epochs), epoch_tick_interval):
        x = padding + i * (width - padding * 2) // (len(epochs) - 1)
        epoch_num = i + 1
        draw.text((x, height - padding + 25), str(epoch_num), font=label_font, fill=text_color)

    # Add legend
    legend_padding = 10
    legend_box_size = 10
    draw.rectangle((width - 200 - legend_padding, padding, width - legend_padding, padding + 40), fill=(255, 255, 255))
    draw.ellipse((width - 190, padding + 10, width - 180, padding + 20), fill=train_color)
    draw.text((width - 170, padding + 5), "Training", font=label_font, fill=text_color)
    draw.ellipse((width - 190, padding + 30, width - 180, padding + 40), fill=val_color)
    draw.text((width - 170, padding + 25), "Validation", font=label_font, fill=text_color)

    # Add title and additional information
    draw.text((width // 2 - 100, 10), "Loss and Accuracy", font=title_font, fill=text_color)
    draw.text((padding, height - padding + 50), f"Final Training Loss: {train_losses[-1]:.3f}", font=label_font, fill=text_color)
    draw.text((padding, height - padding + 70), f"Final Validation Loss: {val_losses[-1]:.3f}", font=label_font, fill=text_color)
    draw.text((padding, height - padding + 90), f"Final Training Accuracy: {train_accuracies[-1]:.2%}", font=label_font, fill=text_color)
    draw.text((padding, height - padding + 110), f"Final Validation Accuracy: {val_accuracies[-1]:.2%}", font=label_font, fill=text_color)

    # Save the image
    img.save(os.path.join(results_path, "loss_and_accuracy.png"))

if __name__ == '__main__':
    main()