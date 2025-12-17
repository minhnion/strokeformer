import torch
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

# --- Bước 1: Định nghĩa đường dẫn và tải file trọng số ---

# Đường dẫn tới file trọng số pre-trained
PRETRAINED_WEIGHT_PATH = './modelBest/selected_layers.pth'

print("="*80)
print("BẮT ĐẦU PHÂN TÍCH FILE TRỌNG SỐ PRE-TRAINED")
print(f"Đang đọc file tại: {PRETRAINED_WEIGHT_PATH}")
print("="*80)

# Kiểm tra xem file có tồn tại không
if not os.path.exists(PRETRAINED_WEIGHT_PATH):
    print(f"Lỗi: Không tìm thấy file trọng số tại '{PRETRAINED_WEIGHT_PATH}'")
    print("Vui lòng đảm bảo bạn đã chạy pre-training hoặc đã đặt file vào đúng vị trí.")
else:
    try:
        # Tải state dictionary. map_location='cpu' đảm bảo code chạy được ngay cả khi không có GPU.
        state_dict = torch.load(PRETRAINED_WEIGHT_PATH, map_location=torch.device('cpu'))
        print("Tải file trọng số pre-trained thành công!\n")

        # --- Bước 2: Kiểm tra thông tin cơ bản ---
        
        print(f"Loại đối tượng được tải: {type(state_dict)}")
        print(f"Tổng số tensors (tham số/lớp) được lưu: {len(state_dict)}\n")

        print("--- Danh sách các lớp và kích thước (shape) của chúng ---")
        # In ra tên và kích thước của mỗi tensor trong state_dict
        for name, params in state_dict.items():
            # Căn chỉnh tên lớp để dễ đọc hơn
            print(f"{name:<70} | Kích thước: {list(params.shape)}")
        print("-"*55)

        # --- Bước 3: Phân tích sâu hơn để suy ra kiến trúc ---
        
        def analyze_architecture(sd):
            """
            Hàm này phân tích state_dict để suy ra các siêu tham số kiến trúc.
            """
            if not sd:
                print("Dictionary trọng số trống.")
                return

            d_token = None
            n_blocks = 0
            d_ffn_hidden = None

            # Lấy thông tin từ tên và kích thước của các lớp
            for name, params in sd.items():
                if 'transformer.blocks.0.attention.W_q.weight' in name:
                    # Kích thước của ma trận W_q là [d_token, d_token]
                    d_token = params.shape[1]
                
                if 'transformer.blocks.' in name:
                    # Tìm số block lớn nhất từ tên lớp, ví dụ: 'transformer.blocks.2. ...'
                    block_num = int(name.split('.')[2])
                    if block_num + 1 > n_blocks:
                        n_blocks = block_num + 1
                
                if 'ffn.linear_first.weight' in name and d_ffn_hidden is None:
                    # Với ReGLU/GEGLU, d_hidden bằng một nửa kích thước đầu ra của lớp linear đầu tiên
                    d_ffn_hidden = params.shape[0] // 2
            
            print("\n--- Phân tích kiến trúc từ trọng số ---")
            if d_token:
                print(f"✅ Kích thước token (d_token): {d_token}")
            if n_blocks:
                print(f"✅ Số lượng khối Transformer (n_blocks): {n_blocks}")
            if d_ffn_hidden:
                print(f"✅ Kích thước ẩn của FFN (d_ffn_hidden): {d_ffn_hidden}")
            
            print("\n(Lưu ý: Số lượng head của attention là một siêu tham số khi tạo mô hình và không thể suy ra trực tiếp từ đây, nhưng nó phải là ước của d_token.)")

        analyze_architecture(state_dict)

        # --- Bước 4: Trực quan hóa phân phối trọng số của một lớp ---
        
        def visualize_weights(sd, layer_name):
            """
            Vẽ biểu đồ histogram phân phối giá trị của một lớp trọng số cụ thể.
            """
            if layer_name in sd:
                # Chuyển tensor sang numpy và làm phẳng để vẽ histogram
                weights = sd[layer_name].numpy().flatten()
                
                print(f"\n--- Trực quan hóa trọng số lớp: {layer_name} ---")
                print(f"Số lượng tham số: {len(weights)}")
                print(f"Giá trị Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")

                plt.figure(figsize=(12, 7))
                plt.hist(weights, bins=100, color='skyblue', edgecolor='black')
                plt.title(f'Phân phối giá trị trọng số cho lớp:\n{layer_name}')
                plt.xlabel('Giá trị')
                plt.ylabel('Tần suất')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.show()
            else:
                print(f"\nLỗi: Không tìm thấy lớp '{layer_name}' trong file trọng số để trực quan hóa.")

        # Chọn một lớp đại diện để vẽ, ví dụ ma trận Query của lớp Attention trong block đầu tiên
        layer_to_visualize = 'transformer.blocks.0.attention.W_q.weight'
        visualize_weights(state_dict, layer_to_visualize)
        
        print("PHÂN TÍCH HOÀN TẤT")
        print("="*80)

    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc hoặc phân tích file: {e}")