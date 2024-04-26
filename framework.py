import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
import timm  
from timm import create_model
from captum.attr import IntegratedGradients
import os
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTFeatureExtractor

class CustomNeuralNetworkModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomNeuralNetworkModel, self).__init__()
        self.swin_transformer = create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=num_classes, features_only=True)
        output_feature_size = 105

        self.reasoning_layer = nn.Sequential(
            nn.Linear(output_feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.swin_transformer(x)
        pooled_features = [F.adaptive_avg_pool2d(feature, (1, 1)).view(feature.size(0), -1) for feature in features]
        concatenated_features = torch.cat(pooled_features, dim=1)
        x = self.reasoning_layer(concatenated_features)
        return x

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512) 
        self.fc2 = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 256 * 28 * 28) 

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def prepare_dataset(data_path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    if not os.path.exists(data_path):
        print(f"The directory {data_path} does not exist.")
        exit()

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    class_names = {v: k for k, v in dataset.class_to_idx.items()}
    
    return dataset, class_names

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def load_model(model_name, num_classes, pretrained=True):
    if model_name == "custom_neural_network":
        model = CustomNeuralNetworkModel(num_classes=num_classes)
    elif model_name == "swin_transformer":
        model = timm.create_model("swin_small_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    elif model_name == "hugging_face_vit":
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_classes)
    elif model_name == "cnn":
        model = CNNModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def train_and_evaluate(model_config, dataset, custom_config):
    model = load_model(**model_config)
    num_classes = model_config["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    default_config = {
        "split_ratio": 0.8,
        "batch_size": 32,
        "epochs": 15,
        "lr": 1e-4,
        "optimizer": "Adam",
        "k_folds": 10,
    }

    config = {**default_config, **custom_config}

    if len(dataset) < 1000 and config["k_folds"] is not None:
            kfold = KFold(n_splits=config["k_folds"], shuffle=True)
            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
                print(f'Fold {fold+1}/{config["k_folds"]}')
                train_subsampler = Subset(dataset, train_ids)
                val_subsampler = Subset(dataset, val_ids)
                train_loader = DataLoader(train_subsampler, batch_size=config["batch_size"], shuffle=True)
                val_loader = DataLoader(val_subsampler, batch_size=config["batch_size"], shuffle=False)
        
                optimizer = getattr(optim, config["optimizer"])(model.parameters(), lr=config["lr"])
                criterion = nn.CrossEntropyLoss()
        
                for epoch in range(config["epochs"]):
                    model.train()
                    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        if hasattr(outputs, 'logits'):
                            outputs = outputs.logits
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
    else:
        train_size = int(len(dataset) *  config["split_ratio"])
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

        optimizer = getattr(optim, config["optimizer"])(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(config["epochs"]):
            model.train()
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
    model.eval()
    y_true = []
    y_pred = []  
    y_score = []  
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            _, predicted = torch.max(outputs, 1)
    
            probabilities = F.softmax(outputs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())  
            y_score.extend(probabilities.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    y_true = label_binarize(y_true, classes=range(num_classes)) 
    y_score = np.array(y_score)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
    print("Training and evaluation completed.")

    return model

def model_wrapper(model, input_tensor):
    outputs = model(input_tensor)
    if hasattr(outputs, 'logits'):
        return outputs.logits 
    else:
        return outputs  

def apply_integrated_gradients(model, input_tensor, target_class, device):
    model.eval()

    def forward_wrapper(inputs):
        return model_wrapper(model, inputs)
        
    integrated_gradients = IntegratedGradients(forward_wrapper)
    attributions = integrated_gradients.attribute(input_tensor, target=target_class, n_steps=200)
    return attributions

def classify(image_path, model, class_names, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    integrated_gradients = IntegratedGradients(model)
    attributions = apply_integrated_gradients(model, input_tensor, top_catid[0], device)
    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.maximum(attributions, 0)
    heatmap = np.sum(attributions, axis=0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap = np.uint8(255 * heatmap)  
    
    heatmap = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR)

    heatmap = np.array(heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)  
    
    overlayed_img = np.array(image) * 0.5 + heatmap * 0.5
    overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlayed_img)
    plt.axis('off')

    plt.tight_layout()
    predicted_class = class_names[top_catid.cpu().item()]
    confidence = top_prob.cpu().item()
    
    plt.show()

    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence)

    return predicted_class, confidence