

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score , recall_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from torchvision import models
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_prepare(Dst, Kls, Size, exts='all'):
    
    if exts == 'all' : exts = ('.jpg', '.jpeg', '.png', '.bmp', )
    else : exts = exts
    
    k_lists = [[] for _ in range(len(Kls))]
    
    for _k in range(len(Kls)):
        k_path = os.path.join(Dst, Kls[_k])
        paths0 = os.listdir(k_path)
        
        for path0 in paths0:
            if path0.endswith(exts):
                imgpath = os.path.join(Dst, Kls[_k], path0)
                img = Image.open(imgpath)
                img = np.array(img.resize((Size, Size))) / 255.0
                img_trans = np.transpose(img, (2, 0, 1))
                k_lists[_k].append(img_trans)
    
    return k_lists

def dataloader(K_lists, Ratio, Batch, mode='train'):
    x, y = [], []
    
    for idx, kl in enumerate(K_lists):
        for k in kl:
            x.append(k)
            y.append(idx)
    
    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    tensor_x, tensor_y = torch.Tensor(x), torch.Tensor(y).long()
    dataset = TensorDataset(tensor_x, tensor_y)
    
    if mode == 'train':
        class_indices = defaultdict(list)
        for idx, label in enumerate(y):
            class_indices[label].append(idx)
        
        tr_indices, val_indices = [], []
        for label, indices in class_indices.items():
            n_train = int(Ratio * len(indices))
            tr_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:])
    
        tr_dataset = torch.utils.data.Subset(dataset, tr_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        tr_loader = DataLoader(tr_dataset, batch_size=Batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Batch, shuffle=False)
        
        return tr_loader, val_loader
    
    elif mode == 'test':
        te_loader = DataLoader(dataset, batch_size=Batch, shuffle=False)
        return te_loader
        
    
def set_model(Kls, Size, model='vgg'):
    if model == 'vgg' : backbone = models.vgg16(pretrained=True)
    elif model == 'resnet' : backbone = models.resnet50(pretrained=True)
    elif model == 'densenet' :backbone = models.densenet121(pretrained=True)
    elif model == 'inception' :backbone = models.inception_v3(pretrained=True)
    elif model == 'mobilenet' :backbone = models.mobilenet_v2(pretrained=True)
    elif model == 'squeezenet' :backbone = models.squeezenet1_0(pretrained=True)
    
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    
    for param in backbone.parameters():
        param.requires_grad = False
    
    _size = backbone(torch.randn(1, *(3, 244, 244))).size()
    
    
    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(int(torch.prod(torch.tensor(_size))), 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(Kls))
    )
    
    Model = nn.Sequential(backbone, head).to(device)
    
    return Model

def train(Tr_loader, Val_loader, Epoch, Lr, Model):

    optimizer = optim.SGD(Model.parameters(), lr=Lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    tl, vl, ta, va = [], [], [], []
    
    result_tr, result_val = [], []
    
    for _ep in range(Epoch):
        Model.train()
        err_tr = 0.0
        correct_tr, total_tr = 0, 0
        
        for batch_tr in tqdm(Tr_loader, desc=f"Epoch {_ep + 1}/{Epoch}"):
            images_tr, labels_tr = batch_tr[0].to(device), batch_tr[1].to(device)
            optimizer.zero_grad()
            
            outputs_tr = Model(images_tr)
            loss_tr = criterion(outputs_tr, labels_tr)
            loss_tr.backward()
            optimizer.step()
            
            err_tr += loss_tr.item()
            _, preds_tr = torch.max(outputs_tr, 1)
            
            total_tr += labels_tr.size(0)
            correct_tr += (preds_tr == labels_tr).sum().item()
            
            result_tr.append([labels_tr.tolist(), preds_tr.tolist()])
        
        tl.append(err_tr / len(Tr_loader))
        ta.append(round(correct_tr / total_tr, 3))
        
        Model.eval()
        err_val = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for batch_val in Val_loader:
                images_val, labels_val = batch_val[0].to(device), batch_val[1].to(device)
                outputs_val = Model(images_val)
                
                loss_val = criterion(outputs_val, labels_val)
                
                err_val += loss_val.item()
                _, preds_val = torch.max(outputs_val, 1)
                
                total_val += labels_val.size(0)
                correct_val += (preds_val == labels_val).sum().item()
                
                
                result_val.append([labels_val.tolist(), preds_val.tolist()])
        
        vl.append(err_val / len(Val_loader))
        va.append(round(correct_val / total_val, 3))
        
        print(f"Epoch {_ep + 1}/{Epoch} - Training Loss: {err_tr / len(Tr_loader):.4f}, "
              f"Validation Loss: {err_val / len(Val_loader):.4f}, Validation Accuracy: {round(correct_val / total_val, 3)}")
    
    plt.figure(figsize=(12, 8))

    # Plotting Loss
    plt.subplot(2, 1, 1)
    plt.plot(range(1, Epoch + 1), tl, marker='o', color='b', label='Train Loss')
    plt.plot(range(1, Epoch + 1), vl, marker='o', color='r', label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(range(1, Epoch + 1), ta, marker='o', color='b', label='Train Accuracy')
    plt.plot(range(1, Epoch + 1), va, marker='o', color='r', label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    
    return Model, result_tr, result_val

def test(Te_loader, Model):
    
    criterion = nn.CrossEntropyLoss()
    
    Model.eval()
    err_te = 0.0
    correct_te = 0
    result_te = []
    pt, pl = [], []
    
    with torch.no_grad():
        for images_te, labels_te in Te_loader:
            images_te, labels_te = images_te.to(device), labels_te.to(device)
            outputs_te = Model(images_te)
            
            loss_te = criterion(outputs_te, labels_te)
            err_te += loss_te.item()
            
            _, preds_te = torch.max(outputs_te, 1)
            
            correct_te += (preds_te == labels_te).sum().item()
            
            result_te.append([labels_te.tolist(), preds_te.tolist()])
            
            pl.append(labels_te[0].item())
            pt.append(preds_te[0].item())
    
    tl = err_te / len(Te_loader)
    ta = round(correct_te / len(Te_loader), 3)

    precision = precision_score(pl, pt, average=None)
    recall = recall_score(pl, pt, average=None)
    
    return result_te, [tl, ta], np.array([precision, recall])





