import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import torch # Thêm import torch để xử lý file torch cũ

# --- Phần 0: Tạo thư mục lưu ảnh ---
IMAGE_SAVE_DIR = './results/visual_outputs'
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)


# --- Phần 1: Các hàm trực quan hóa (Đã được cập nhật) ---

def plot_roc_curve(y_true, y_pred_proba, dataset_name):
    """
    Vẽ đường cong ROC và tính diện tích AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Đường cong ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính Giả (False Positive Rate)')
    plt.ylabel('Tỷ lệ Dương tính Thật (True Positive Rate)')
    plt.title(f'Đường cong ROC cho bộ dữ liệu: {dataset_name}')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # THAY THẾ plt.show() BẰNG plt.savefig()
    save_path = os.path.join(IMAGE_SAVE_DIR, f'{dataset_name}_roc_curve.png')
    plt.savefig(save_path)
    plt.close() # Đóng biểu đồ để giải phóng bộ nhớ
    print(f"Đã lưu biểu đồ ROC tại: {save_path}")


def plot_precision_recall_curve(y_true, y_pred_proba, dataset_name):
    """
    Vẽ đường cong Precision-Recall và tính AP.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, where='post', color='b', alpha=0.7, label=f'Đường cong P-R (AP = {avg_precision:.4f})')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='b')
    plt.xlabel('Recall (Độ nhạy)')
    plt.ylabel('Precision (Độ chính xác)')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Đường cong Precision-Recall cho bộ dữ liệu: {dataset_name}')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # THAY THẾ plt.show() BẰNG plt.savefig()
    save_path = os.path.join(IMAGE_SAVE_DIR, f'{dataset_name}_pr_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu biểu đồ Precision-Recall tại: {save_path}")


def plot_confusion_matrix(y_true, y_pred_proba, dataset_name, threshold=0.5):
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix).
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Dự đoán 0 (Không đột quỵ)', 'Dự đoán 1 (Có đột quỵ)'],
                yticklabels=['Thực tế 0 (Không đột quỵ)', 'Thực tế 1 (Có đột quỵ)'])
    plt.title(f'Ma trận nhầm lẫn cho: {dataset_name} (Ngưỡng = {threshold})')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    
    # THAY THẾ plt.show() BẰNG plt.savefig()
    save_path = os.path.join(IMAGE_SAVE_DIR, f'{dataset_name}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu ma trận nhầm lẫn tại: {save_path}")

def plot_training_history(log_dir):
    """
    Đọc dữ liệu từ thư mục log của TensorBoard và vẽ đồ thị loss.
    Hàm này sẽ vẽ loss của tất cả các fold và repeat trên cùng một đồ thị.
    """
    if not os.path.exists(log_dir) or not os.listdir(log_dir):
        print(f"Thư mục log '{log_dir}' không tồn tại hoặc trống.")
        return

    # Lấy danh sách các file event trong thư mục
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'tfevents' in f]
    
    if not event_files:
        print(f"Không tìm thấy file event của TensorBoard trong '{log_dir}'.")
        return

    plt.figure(figsize=(15, 9))
    
    all_tags_found = False
    for event_file in event_files:
        try:
            # Khởi tạo accumulator để đọc file event
            ea = event_accumulator.EventAccumulator(event_file,
                size_guidance={event_accumulator.SCALARS: 0})
            ea.Reload() # Tải dữ liệu

            # Lấy tất cả các tag (ví dụ: '0_0_Training loss', '0_0_Validation loss')
            tags = ea.Tags()['scalars']
            if tags:
                all_tags_found = True
            
            for tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                # Lấy tên của fold/repeat từ tag để làm nhãn cho đường cong
                label_name = tag.replace('_Training loss', ' Train').replace('_Validation loss', ' Val')
                
                if 'Val' in label_name:
                    plt.plot(steps, values, linestyle='--', alpha=0.7) # Bỏ label để legend không quá rối
                else:
                    plt.plot(steps, values, alpha=0.5) # Bỏ label để legend không quá rối

        except Exception as e:
            print(f"Lỗi khi đọc file {event_file}: {e}")

    if not all_tags_found:
        print("Không tìm thấy dữ liệu scalar (loss) nào trong các file event.")
        plt.close()
        return
        
    # Tạo legend tùy chỉnh để không bị quá tải
    plt.plot([], [], color='blue', alpha=0.5, label='Training Loss (các fold)')
    plt.plot([], [], color='orange', linestyle='--', alpha=0.7, label='Validation Loss (các fold)')

    plt.title('Lịch sử Training & Validation Loss (Tất cả các Fold/Repeat)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCEWithLogitsLoss)')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # THAY THẾ plt.show() BẰNG plt.savefig()
    save_path = os.path.join(IMAGE_SAVE_DIR, 'training_history.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu lịch sử huấn luyện tại: {save_path}")


# --- Phần 2: Hàm chính để thực thi ---

def main(dataset_name):
    print("="*80)
    print(f"BẮT ĐẦU TRỰC QUAN HÓA KẾT QUẢ CHO BỘ DỮ LIỆU: {dataset_name}")
    print("="*80)

    # --- 2.1 Trực quan hóa kết quả dự đoán ---
    label_file_npy = f'./visualResult/ROCData/FI-Transformer-Noise_{dataset_name}_labels.npy'
    pred_file_npy = f'./visualResult/ROCData/FI-Transformer-Noise_{dataset_name}_preds.npy'

    # Để tương thích với các file cũ hơn được lưu bằng torch từ evalCommon.py
    label_file_torch = f"./Strokeformer3_label1"
    pred_file_torch = f"./Strokeformer3_pred1"

    y_true, y_pred_proba = None, None

    if os.path.exists(label_file_npy) and os.path.exists(pred_file_npy):
        print(f"Tìm thấy file kết quả .npy cho '{dataset_name}'. Đang tải...")
        y_true = np.load(label_file_npy)
        y_pred_proba = np.load(pred_file_npy)
    elif os.path.exists(label_file_torch) and os.path.exists(pred_file_torch):
        print(f"Tìm thấy file kết quả torch cho '{dataset_name}'. Đang tải...")
        y_true = torch.load(label_file_torch, weights_only=False)
        y_pred_proba = torch.load(pred_file_torch, weights_only=False)
        # Chuyển đổi sang numpy nếu là tensor
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        if isinstance(y_pred_proba, torch.Tensor):
            y_pred_proba = y_pred_proba.numpy()
    else:
        print(f"Lỗi: Không tìm thấy file kết quả (.npy hoặc torch) cho '{dataset_name}'.")
        print("Vui lòng chạy lại main.py và đảm bảo đã bỏ comment các dòng np.save hoặc file được lưu từ evalCommon.")
        return # Thoát nếu không có dữ liệu

    if y_true is not None and y_pred_proba is not None:
        print("\n--- Đang vẽ các biểu đồ đánh giá ---")
        plot_roc_curve(y_true, y_pred_proba, dataset_name)
        plot_precision_recall_curve(y_true, y_pred_proba, dataset_name)
        plot_confusion_matrix(y_true, y_pred_proba, dataset_name)
    
    # --- 2.2 Trực quan hóa lịch sử huấn luyện ---
    log_directory = "./losscurve/FITransformer_loss"
    print(f"\n--- Đang vẽ lịch sử huấn luyện từ thư mục: {log_directory} ---")
    plot_training_history(log_directory)
    
    print("TRỰC QUAN HÓA HOÀN TẤT")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trực quan hóa kết quả huấn luyện và đánh giá mô hình.")
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help='Tên bộ dữ liệu đã chạy (ví dụ: StrokeHealth, Blood, Hospital2)')
    
    args = parser.parse_args()
    main(args.dataset_name)