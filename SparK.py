import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm

# Load the Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Oxford IIIT Pet dataset does not directly support a separate target_transform for masks, so ensure the resize operation is consistent.
dataset = datasets.OxfordIIITPet(root="data", split='trainval', download=True, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Load the pretrained model
model = timm.create_model('resnet50', pretrained=False, num_classes=0)  # num_classes=0 to remove the final FC layer
model.load_state_dict(torch.load("resnet50_1kpretrained_timm_style.pth"), strict=False)

# Fine tune the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
print("start training")
for epoch in range(10):  # Number of epochs
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:  # Assuming masks are directly the second element
        images, masks = images.to(device), masks.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)  # Direct use without assuming a dictionary

        # Ensure outputs and masks are correctly aligned; might need to adjust dimensions
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")