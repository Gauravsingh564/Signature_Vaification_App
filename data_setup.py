import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def extract_images_from_subdirectories(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for subdir, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                shutil.copy2(os.path.join(subdir, file),
                             os.path.join(destination_folder, file))

def get_data_loaders(train_dir, test_dir, img_size=(224,224),
                     batch_size=32, num_workers=4, pin_memory=True):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory)
    return train_loader, test_loader, train_dataset, test_dataset

def visualize_predictions(model, loader, class_names, device, num_images=6):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    preds = outputs.argmax(dim=1)
    images = images.cpu().numpy().transpose((0,2,3,1))
    fig, axes = plt.subplots(1, num_images, figsize=(12,4))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow((images[i]*0.5 + 0.5))
        ax.axis('off')
        ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
    plt.show()
