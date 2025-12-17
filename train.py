import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from focal_loss import FocalLoss

loss_path = "./losscurve/FITransformer_loss"
if not os.path.exists(loss_path):
    os.makedirs(loss_path)

writer = SummaryWriter(loss_path)


class FITransformer_trainer():
    def __init__(self, model, train_dataloader, val_dataloader, num_epochs=500, learning_rate=1e-3, pos_weight_value=1.0, alpha=0.25, gamma=2.0, num=0, repeat=0, device=""):
        super().__init__()
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        # print(f"Initializing loss with pos_weight: {pos_weight_value:.4f}")
        # pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
       
        print(f"Initializing with Focal Loss (alpha={alpha:.4f}, gamma={gamma:.4f})")
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-5)

        self.early_stopping = 0
        self.patience = 5
        self.best_loss = float('inf')
        self.device = device
        self.num = num
        self.repeat = repeat

    def train(self):
        for epoch in range(self.num_epochs):
            # train
            self.model.train()
            total_train_loss = 0
            for num, cat, labels in tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.num_epochs}"):
                num, cat, labels = num.to(self.device), cat.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                # 正常输出
                outputs = self.model(num, cat)
                outputs = torch.squeeze(outputs)

                # 正常计算loss
                loss = self.criterion(outputs, labels)

                total_train_loss += loss.item()
                loss.backward()

            
                self.optimizer.step()
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            writer.add_scalar(f'{self.repeat}_{self.num}_Training loss', avg_train_loss, epoch)
        
            # val
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():  # no grad
                for num, cat, labels in self.val_dataloader:
                    num, cat, labels = num.to(self.device), cat.to(self.device), labels.to(self.device)
                    outputs = self.model(num, cat)
                    outputs = torch.squeeze(outputs)
                    loss = self.criterion(outputs, labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(self.val_dataloader)
            writer.add_scalar(f'{self.repeat}_{self.num}_Validation loss', avg_val_loss, epoch)

            # early stopping
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                torch.save(self.model.state_dict(), f'./modelBest/FITransformer_{self.num}_modelBestParameters')
                self.early_stopping = 0
            else:
                self.early_stopping += 1
                if self.early_stopping >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break  
        writer.close()