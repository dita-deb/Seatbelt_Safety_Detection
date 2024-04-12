import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# Constants
DATA_PATH = "/path/to/your/data/"
CLASSES = ["safe driving", "texting - right", "talking on the phone - right", "texting - left", 
           "talking on the phone - left", "operating the radio", "drinking", "reaching behind", 
           "hair and makeup", "talking to passenger"]

# Hyperparameters
batch_size = 64
num_epochs = 10
resnet_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmented transforms
augmented_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(resnet_size, scale=(0.75, 1.0)),
    transforms.ToTensor(),
])

# Load labeled dataset
data = datasets.ImageFolder(DATA_PATH, transform=augmented_transforms)
num_train = len(data)
indices = list(range(num_train))
split = int(np.floor(0.8 * num_train))
np.random.seed(4747)
np.random.shuffle(indices)
train_idx, valid_idx = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
dataset_sizes = {'train': split, 'val': num_train - split}
train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler)
dataloaders = {'train': train_loader, 'val': valid_loader}

# Model training function
def train_model(model, criterion, optimizer, num_epochs):
    history = []
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0
        valid_acc = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            predictions = outputs.argmax(dim=1, keepdim=True)
            comparisons = predictions.eq(labels.view_as(predictions))
            total += labels.shape[0]
            correct += int(comparisons.double().sum().item())
            loss.backward()
            optimizer.step()
        train_acc = correct / total
        train_loss = train_loss / total
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                predictions = outputs.argmax(dim=1, keepdim=True)
                comparisons = predictions.eq(labels.view_as(predictions))
                total += labels.shape[0]
                correct += int(comparisons.double().sum().item())
            valid_acc = correct / total
            valid_loss = valid_loss / total
        history.append([train_loss, valid_loss, train_acc, valid_acc])
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")
    print("Finished Training")
    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history

# Model evaluation function
def eval_model(model):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    class_correct = [0 for _ in CLASSES]
    class_total = [0 for _ in CLASSES]
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1, keepdim=True)
            comparisons = predictions.eq(labels.view_as(predictions))
            for comp, label in zip(comparisons, labels):
                class_correct[label] += comp.item()
                class_total[label] += 1
            total += labels.shape[0]
            correct += int(comparisons.double().sum().item())
    accuracy = correct / total
    print(f"Accuracy on validation set: {accuracy*100:.2f}%")
    for i, cls in enumerate(CLASSES):
        ccorrect = class_correct[i]
        ctotal = class_total[i]
        caccuracy = ccorrect / ctotal
        print(f"  Accuracy on {cls}: {caccuracy*100:.2f}%")

# Visualization function
def visualize_model(model):
    model.eval()
    model.to(device)
    images, labels = next(iter(valid_loader))
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1, keepdim=True)
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        ax = plt.subplot(4, 4, i + 1)
        ax.axis('off')
        ax.set_title(f'Predicted: {CLASSES[predictions[i].item()]}\nActual: {CLASSES[labels[i].item()]}')
        plt.imshow(images[i].cpu().permute(1, 2, 0))
    plt.show()

# Confusion matrix plotting function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict
