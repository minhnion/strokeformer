import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


loss_path = "./losscurve/pre_FITransformer_loss"
if not os.path.exists(loss_path):
    os.makedirs(loss_path)

writer = SummaryWriter(loss_path)


class Pretrain_FITransformer_trainer():
    def __init__(self, model, train_dataloader, data_noise, num_epochs=500, learning_rate=1e-3, num=0, repeat=0, device=""):
        super().__init__()
        self.model = model
        self.train_dataloader = train_dataloader
        self.data_noise = data_noise
        self.num_epochs = num_epochs
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.device = device

    def train(self):
        for epoch in range(self.num_epochs):
            # train
            self.model.train()
            total_train_loss = 0
            for num, cat in tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.num_epochs}"):
                num, cat = num.to(self.device), cat.to(self.device)

                # 对特征进行noise的操作
                corrupt_feature, mask = self.data_noise.apply(torch.cat([num, cat], dim=-1))
                corrupt_num, corrupt_cat = torch.split(corrupt_feature, [num.size(-1), cat.size(-1)], dim=-1)

                corrupt_cat = corrupt_cat.long()
                # 正常输出
                self.optimizer.zero_grad()
                reconstruct_feature, predicted_mask = self.model(corrupt_num, corrupt_cat)
            
                # 计算loss，预训练模型的关键部分
                loss = self.model.loss(num, cat, reconstruct_feature, mask, predicted_mask)

                total_train_loss += loss.item()
                loss.backward()

            
                self.optimizer.step()
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            writer.add_scalar(f'Training loss', avg_train_loss, epoch)
        
        
        layers_to_save = ["transformer"]
        # 从 state_dict 中筛选指定层的参数
        selected_state_dict = {k: v for k, v in self.model.state_dict().items() if any(layer in k for layer in layers_to_save)}

        # 保存选定层的参数
        torch.save(selected_state_dict, './modelBest/selected_layers.pth')
       