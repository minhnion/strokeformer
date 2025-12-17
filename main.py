
import torch
import numpy as np
import random
import argparse

from FITransformer_pretrain_model import Pretrain_FITransformer, NoiseMasker
from FITransformer_model import FITransformer

from data.covertype_dataset import CovertypeDataset
from data.Hospital_dataset import HospitalDataset
from data.Hospital2_dataset import HospitalDataset2

from data.IST3_dataset import IST3Dataset
from data.Dresses_dataset import DressesDataset
from data.Blood_dataset import BloodDataset
from data.Diabetes_dataset import DiabetesDataset
from data.Ilpd_dataset import IlpdDataset
from data.Breastw_dataset import BreastwDataset
from data.CreditA_dataset import CreditADataset
from data.Kc2_dataset import Kc2Dataset
from data.Qsar_dataset import QsarDataset
from data.Tic_dataset import TicDataset
from data.Wdbc_dataset import WdbcDataset
from data.StrokeHealth_dataset import StrokeHealthDataset

from pretrain import Pretrain_FITransformer_trainer
from train import FITransformer_trainer

from evalCommon import evalCommon
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, random_split


def main(train_mode, downstream_name): 
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)

    batch_size = 256
    learning_rate = 2 * 1e-4
    if train_mode == "pretrain":
        # 预训练
        # 测试用文件
        # pretrain_dataset = CovertypeDataset(path_file="./data/test_covertype.csv")

        pretrain_dataset = CovertypeDataset(path_file="./data/covertype.csv")
        pretrain_num_continuous = 10
        pretrain_cat_card = [2] * 44
        pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

        # 数据噪声
        probs = 0.5
        noise_generate = NoiseMasker(probs)
        # 模型
        pretrain_model = Pretrain_FITransformer.make_default(
                        n_num_features=pretrain_num_continuous,
                        cat_cardinalities=pretrain_cat_card,
                        d_out=1,
                    )
        pretrain_model.to(device)

        # 预训练过程
        Pretrain_FITransformer_trainer(pretrain_model, pretrain_dataloader,
                                       noise_generate, num_epochs=30, learning_rate=learning_rate, device=device).train()


    # 微调
    if downstream_name == "Hospital":
        # 加载数据集
        downstream_dataset = HospitalDataset(path_file="./data/raw_data.xlsx")
        downstream_num_continuous = 2
        downstream_cat_card = [4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    elif downstream_name == "Hospital2":
        # 加载数据集
        downstream_dataset = HospitalDataset2(path_file="./data/results.xlsx")
        downstream_num_continuous = 6
        downstream_cat_card = [2, 2, 2, 2, 2, 2, 6, 4]
    elif downstream_name == "IST3":
        downstream_dataset = IST3Dataset(path_file="./data/datashare_aug2015.csv")
        downstream_num_continuous = 6
        downstream_cat_card = [3, 5, 2, 2, 2, 2, 2, 3] 
    elif downstream_name == "Blood":
        downstream_dataset = BloodDataset(path_file="./data/dataset_1464.csv")
        downstream_num_continuous = 4
        downstream_cat_card = []
    elif downstream_name == "Breastw":
        downstream_dataset = BreastwDataset(path_file="./data/dataset_15.csv")
        downstream_num_continuous = 9
        downstream_cat_card = []
    elif downstream_name == "CreditA":
        downstream_dataset = CreditADataset(path_file="./data/dataset_29.csv")
        downstream_num_continuous = 6
        downstream_cat_card = [2,3,3,14,9,2,2,2,3]
    elif downstream_name == "Diabetes":
        downstream_dataset = DiabetesDataset(path_file="./data/dataset_37.csv")
        downstream_num_continuous = 8
        downstream_cat_card = []
    elif downstream_name == "Dresses":
        downstream_dataset = DressesDataset(path_file="./data/dataset_23381.csv")
        downstream_num_continuous = 1
        downstream_cat_card = [13, 7, 7, 8, 16, 17, 4, 23, 22, 24, 14]
    elif downstream_name == "Ilpd":
        downstream_dataset = IlpdDataset(path_file="./data/dataset_1480.csv")
        downstream_num_continuous = 9
        downstream_cat_card = [2]
    elif downstream_name == "Kc2":
        downstream_dataset = Kc2Dataset(path_file="./data/dataset_1063.csv")
        downstream_num_continuous = 21
        downstream_cat_card = []
    elif downstream_name == "Qsar":
        downstream_dataset = QsarDataset(path_file="./data/dataset_1494.csv")
        downstream_num_continuous = 41
        downstream_cat_card = []
    elif downstream_name == "Tic":
        downstream_dataset = TicDataset(path_file="./data/dataset_50.csv")
        downstream_num_continuous = 0
        downstream_cat_card = [3,3,3,3,3,3,3,3,3]
    elif downstream_name == "Wdbc":
        downstream_dataset = WdbcDataset(path_file="./data/dataset_1510.csv")
        downstream_num_continuous = 30
        downstream_cat_card = []
    elif downstream_name == "StrokeHealth":
        downstream_dataset = StrokeHealthDataset(path_file="./data/synthetic_multimodal.csv")
        downstream_num_continuous = 3
        # gender(2), hypertension(2), heart_disease(2), ever_married(2), work_type(5), Residence_type(2), smoking_status(4)
        downstream_cat_card = [2, 2, 2, 2, 5, 2, 4]

    # 交叉验证的方式

    # 10折交叉验证
    n_splits = 10
    # 重复3次 
    repeats = 3
    # K折交叉验证数据集划分
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    train_loaders = []
    val_loaders = []
    test_loaders = []

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(downstream_dataset)):
        train_val_subset = Subset(downstream_dataset, train_val_idx)
        test_subset = Subset(downstream_dataset, test_idx)

        train_subset, val_subset = random_split(train_val_subset, [len(train_val_subset) - int(len(train_val_subset) * 0.2), int(len(train_val_subset) * 0.2)])
        # 创建DataLoader
        train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
    
    # 3次10折交叉验证的平均评估指标
    total_metrics = {
        'accuracy': 0,
        'recall' : 0,
        'precision':0,
        'f1':0,
        'roc_auc': 0,
    }
    all_repeat_labels = []
    all_repeat_preds = []

    for repeat in range(repeats):
        # 汇总10折交叉验证模型的结果和标签
        all_labels = []
        all_preds = []
        for idx in range(n_splits):
            print(f"Repeat: {repeat} Model {idx+1}------------------------------")
            
            
            # 读取预训练模型的参数并初始化微调模型
            finue_model = FITransformer.make_default(
                            n_num_features=downstream_num_continuous,
                            cat_cardinalities=downstream_cat_card,
                            d_out=1,
            )
            finue_model.load_state_dict(
                torch.load('./modelBest/selected_layers.pth', map_location=torch.device('cpu')), strict=False)
            finue_model.to(device)

            # 冻结Transformer层的参数
            # for param in finue_model.transformer.parameters():
            #     param.requires_grad = False
            
            FITransformer_trainer(finue_model, train_loaders[idx], val_loaders[idx], num_epochs=2000, learning_rate= 3 * 1e-4, num=idx, repeat=repeat, device=device).train()
            finue_model.load_state_dict(torch.load(f'./modelBest/FITransformer_{idx}_modelBestParameters'))
        
            # 预测
            finue_model.eval()

            preds = []
            labels = []
            for num, cat, label in test_loader:
                num, cat, label = num.to(device), cat.to(device), label.to(device)
                outputs = finue_model(num, cat)

                outputs = torch.sigmoid(outputs)
                preds.append(outputs.detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            
            all_labels.append(labels)
            all_preds.append(preds)

        # 将统计的预测结果和标签拼接在一起
        all_labels_np = np.concatenate(all_labels)
        all_preds_np = np.concatenate(all_preds)

        # 对于10折的评估指标
        new_metrics = evalCommon(all_labels_np, all_preds_np).evaluation()
        for key, value in total_metrics.items():
            total_metrics[key] += new_metrics[key]


        all_repeat_labels.append(all_labels_np)
        all_repeat_preds.append(all_preds_np)
    
    # 平均的准确率和loss
    total_average_metrics = {k : v / repeats for k, v in total_metrics.items()}
    for metric, value in total_average_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    
    all_repeat_labels = np.concatenate(all_repeat_labels)
    all_repeat_preds = np.concatenate(all_repeat_preds)

    # np.save(f'./visualResult/ROCData/FI-Transformer-Noise_{downstream_name}_labels.npy', all_repeat_labels)
    # np.save(f'./visualResult/ROCData/FI-Transformer-Noise_{downstream_name}_preds.npy', all_repeat_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default='no pretrain', help='Type of train')
    parser.add_argument('--downstream_name', type=str, default='Hospital', help='Type of downstream dataset')
    args = parser.parse_args()
    main(args.train_mode, args.downstream_name)
