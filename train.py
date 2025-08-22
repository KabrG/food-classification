import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import Food101
import torchvision.models as models

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


# Create a transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # default mean and median
    transforms.Resize((224, 224)) # Make it smaller to improve the training speed

])

# Grab the datasets
train_dataset = datasets.Food101(root="./data", split="train", download=True, transform=transform)
test_dataset = datasets.Food101(root="./data", split="test", download=True, transform=transform)


# Initialize dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Shuffle is false to get consistent testing results

# Display image for fun
# for batch in train_loader:
#     img, label = batch
#     img = img[0]
#     print(train_dataset.classes[label[0]]) # Mapping to the actual name 
#     print(img.shape)
#     import matplotlib.pyplot as plt; plt.imshow(img.permute(1,2,0).numpy()); plt.axis('off'); plt.show()

#     exit()


# Define the model. Have it pretrained
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Define optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()


# Try to load existing model if it exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device) 

try:
    model.load_state_dict(torch.load("food_detector_model.pth", map_location=device))
    print("Loaded existing weights.")
except FileNotFoundError:
    print("No existing weights found. Training from scratch.")


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
          f" ({accuracy:.2f}%)\n")


# Training loop

print("Testing prior to training")

epochs = 1
test()
for epoch in range(1, epochs): 
    train(epoch)
    test()

    # Open file in read mode and read if it should continue running
    run_status = "stop" # By default, the program will stop
    try:
        with open("continue_running.txt", "r") as file:
            run_status = str(file.read()).lower()

    except Exception as p:
        print(p)

    if run_status == "stop":
        print("Halting training...")
        break
    else:
        print("Continue training")
        pass

torch.save(model.state_dict(), "food_detector_model.pth")
print("Model weights saved to food_detector_model.pth")

