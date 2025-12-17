import torch
import numpy as np
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
from datetime import datetime

# Import các thành phần của Strokeformer
from FITransformer_model import FITransformer
from data.StrokeHealth_dataset import StrokeHealthDataset 
from train import FITransformer_trainer
from evalCommon import evalCommon
from torch.utils.data import DataLoader

# === LỚP LOGGER ĐÃ SỬA LỖI ===
class Logger(object):
    """
    Lớp Logger để ghi output ra cả terminal và file.
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding='utf-8') # Mở file với encoding utf-8

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() # **QUAN TRỌNG: Xả bộ đệm ngay sau khi ghi**

    def flush(self):
        # **QUAN TRỌNG: Thực sự gọi lệnh flush**
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # Hàm để đóng file log
        self.log.close()

def run_strokeformer_experiment():
    # --- 1. CÀI ĐẶT CẤU HÌNH ---
    print("="*80)
    print("BẮT ĐẦU THỬ NGHIỆM STROKEFORMER")
    print("="*80)
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    

    raw_data_path = './data/healthcare-dataset-stroke-data.csv'
    drop_columns = ["id"]
    target_col = "stroke"
    categorical_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]
    numerical_features = ["age", "avg_glucose_level", "bmi"]
    split_params = {'test_size': 0.2, 'val_size': 0.1, 'random_state': 42}
    
    # --- 2. TIỀN XỬ LÝ DỮ LIỆU ---
    print("\n--- Bước 1: Tiền xử lý dữ liệu ---")
    df = pd.read_csv(raw_data_path)
    df = df.drop(columns=drop_columns)
    
    df['bmi'] = df['bmi'].replace('N/A', np.nan)
    df['bmi'] = pd.to_numeric(df['bmi'])
    df['bmi'].fillna(df['bmi'].median(), inplace=True)
    df = df[df['gender'] != 'Other']

    cat_cardinalities = []
    for col in categorical_features:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        cat_cardinalities.append(len(encoder.classes_))
    
    print(f"Dữ liệu đã được làm sạch. Tổng số mẫu: {len(df)}")
    print(f"Cardinalities của các cột phân loại: {cat_cardinalities}")

    # --- 3. CHIA DỮ LIỆU ---
    print("\n--- Bước 2: Chia dữ liệu Train/Validation/Test ---")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=split_params['test_size'],
        random_state=split_params['random_state'],
        stratify=y
    )
    
    val_size_adjusted = split_params['val_size'] / (1 - split_params['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        random_state=split_params['random_state'],
        stratify=y_train_val
    )
    print(f"Kích thước tập Train: {len(X_train)}")
    print(f"Kích thước tập Validation: {len(X_val)}")
    print(f"Kích thước tập Test: {len(X_test)}")

    # --- 4. CHUẨN HÓA DỮ LIỆU ---
    print("\n--- Bước 3: Chuẩn hóa dữ liệu (Standardization) ---")
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_val[numerical_features] = scaler.transform(X_val[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    print("Đã chuẩn hóa các đặc trưng số.")

    # --- 5. TẠO DATASET VÀ DATALOADER ---
    train_dataset = StrokeHealthDataset(X_train, y_train, categorical_features, numerical_features)
    val_dataset = StrokeHealthDataset(X_val, y_val, categorical_features, numerical_features)
    test_dataset = StrokeHealthDataset(X_test, y_test, categorical_features, numerical_features)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- 6. KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH ---
    print("\n--- Bước 4: Khởi tạo và Huấn luyện mô hình Strokeformer ---")
    model = FITransformer.make_default(
        n_num_features=len(numerical_features),
        cat_cardinalities=cat_cardinalities,
        d_out=1,
    )
    model.load_state_dict(
        torch.load('./modelBest/selected_layers.pth', map_location=device), strict=False)
    model.to(device)

    print("Model initialized and pre-trained weights loaded.")
    print("Chế độ: Fine-tuning toàn bộ mô hình (end-to-end).")

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight_value = (neg_count / pos_count) if pos_count > 0 else 1.0

    trainer = FITransformer_trainer(
        model=model, 
        train_dataloader=train_loader, 
        val_dataloader=val_loader, 
        num_epochs=200, 
        learning_rate=1e-4, 
        pos_weight_value=pos_weight_value,
        num=0, 
        repeat=0,
        device=device
    )
    trainer.train()

    # --- 7. ĐÁNH GIÁ TRÊN TẬP TEST ---
    print("\n--- Bước 5: Đánh giá trên tập Test ---")
    best_model_path = f'./modelBest/FITransformer_0_modelBestParameters'
    if not os.path.exists(best_model_path):
         print(f"Lỗi: Không tìm thấy file trọng số tốt nhất tại {best_model_path}")
         return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for num, cat, label in test_loader:
            num, cat, label = num.to(device), cat.to(device), label.to(device)
            outputs = model(num, cat)
            preds.append(torch.sigmoid(outputs).cpu().numpy())
            labels.append(label.cpu().numpy())
    
    all_preds_np = np.concatenate(preds).flatten()
    all_labels_np = np.concatenate(labels).flatten()

    # --- 8. TÍNH TOÁN VÀ IN KẾT QUẢ ---
    final_metrics = evalCommon(all_labels_np, all_preds_np).evaluation()
    print("\n--- KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST ---")
    for metric, value in final_metrics.items():
        print(f"{metric.capitalize():<10}: {value:.4f}")
    
    print("THỬ NGHIỆM HOÀN TẤT")
    print("="*80)

# === ĐIỂM BẮT ĐẦU CHẠY SCRIPT (CẤU TRÚC MỚI) ===
if __name__ == "__main__":
    # Thiết lập logger
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"strokeformer_experiment_{timestamp}.txt")
    
    original_stdout = sys.stdout # Lưu lại stdout gốc
    logger = Logger(log_file_path)
    sys.stdout = logger

    try:
        run_strokeformer_experiment() # Gọi hàm chính
    except Exception as e:
        print(f"\nĐÃ XẢY RA LỖI: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Khối finally này sẽ LUÔN LUÔN được thực thi, dù có lỗi hay không
        sys.stdout = original_stdout # Trả lại stdout gốc
        logger.close() # Đóng file log
        print(f"\nLog đã được lưu vào file: {log_file_path}")