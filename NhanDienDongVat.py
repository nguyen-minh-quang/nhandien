import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import os
from translate import translate

# Kích thước khung ảnh và kích thước đầu vào của mô hình
IMG_WIDTH = 250
IMG_HEIGHT = 250
MODEL_IMG_SIZE = (150, 150)

# Biến toàn cục để lưu trữ ảnh gốc, mô hình và tên các lớp
current_img = None
model = None
class_names = []

# Tạo từ điển dịch từ tiếng Anh sang tiếng Việt
# Đây là phần quan trọng để hiển thị kết quả bằng tiếng Việt
translate_to_vietnamese = {
    "dog": "Chó",
    "horse": "Ngựa",
    "elephant": "Voi",
    "butterfly": "Bướm",
    "chicken": "Gà",
    "cat": "Mèo",
    "cow": "Bò",
    "sheep": "Cừu",
    "squirrel": "Sóc",
    "spider": "Nhện"
}


# Hàm tải mô hình và tên các lớp từ dữ liệu đã huấn luyện
def load_model_and_classes():
    global model, class_names
    try:
        # Tải mô hình đã huấn luyện được lưu dưới tên 'best_model.h5'
        model = tf.keras.models.load_model('best_model.h5')

        # Tải danh sách tên các lớp gốc (tiếng Ý) từ thư mục dữ liệu
        base_dir = "dataset/raw-img/"
        class_names_raw = sorted(os.listdir(base_dir))
        class_names_raw = [d for d in class_names_raw if os.path.isdir(os.path.join(base_dir, d))]

        # Dịch tên các lớp từ tiếng Ý sang tiếng Anh, sau đó sang tiếng Việt
        class_names = [translate_to_vietnamese[translate[name]] for name in class_names_raw]

        print("Mô hình và tên các lớp đã được tải thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình hoặc tên lớp: {e}")
        model = None
        class_names = []


# Hàm tải ảnh gốc và hiển thị lên giao diện
def load_image():
    global current_img
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        img = Image.open(file_path)
        w, h = img.size
        ratio = min(IMG_WIDTH / w, IMG_HEIGHT / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        current_img = img
        img_tk = ImageTk.PhotoImage(img)
        lbl_image_left.config(image=img_tk, text="")
        lbl_image_left.image = img_tk

        lbl_image_right.config(image="", text="Ảnh Kết quả")
        lbl_result_text.config(text="Kết quả dự đoán...")


# Hàm dự đoán động vật khi nhấn nút
def predict_animal():
    global current_img, model, class_names
    if current_img is None:
        lbl_result_text.config(text="⚠ Vui lòng chọn ảnh trước")
        return
    if model is None:
        lbl_result_text.config(text="⚠ Lỗi: Không thể tải mô hình")
        return

    img_np = np.array(current_img.convert('RGB'))
    img_resized = cv2.resize(img_np, MODEL_IMG_SIZE)
    img_processed = img_resized.astype("float32") / 255.0

    img_input = np.expand_dims(img_processed, axis=0)

    # Thực hiện dự đoán
    predictions = model.predict(img_input)[0]
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[predicted_class_index] * 100

    # Hiển thị ảnh dự đoán (sử dụng lại ảnh đã tải)
    img_tk = ImageTk.PhotoImage(current_img)
    lbl_image_right.config(image=img_tk, text="")
    lbl_image_right.image = img_tk

    # Lấy tên động vật bằng tiếng Việt từ danh sách đã dịch
    predicted_animal = class_names[predicted_class_index]

    # Cập nhật label để hiển thị kết quả
    lbl_result_text.config(text=f"Kết quả dự đoán: {predicted_animal}\nĐộ tin cậy: {confidence:.2f}%")


# =================================================================
# Tạo cửa sổ chính và các thành phần giao diện
# =================================================================
root = tk.Tk()
root.title("Nhận diện động vật")
root.geometry("900x600")

# Tải mô hình khi khởi động ứng dụng
load_model_and_classes()

# ========== TIÊU ĐỀ ==========
title = tk.Label(root, text="TRƯỜNG ĐẠI HỌC CÔNG NGHỆ ĐÔNG Á\nKHOA CÔNG NGHỆ THÔNG TIN",
                 font=("Arial", 14, "bold"), justify="center")
title.pack(pady=10)

# ========== FRAME CHÍNH ==========
frame_main = tk.Frame(root)
frame_main.pack(pady=10, fill="both", expand=True)

# ---- Khung trái: Ảnh gốc ----
frame_left = tk.Frame(frame_main)
frame_left.pack(side="left", padx=30)
box_left = tk.Frame(frame_left, bd=2, relief="groove", width=IMG_WIDTH, height=IMG_HEIGHT)
box_left.pack()
box_left.pack_propagate(False)
lbl_image_left = tk.Label(box_left, text="Ảnh Gốc")
lbl_image_left.pack(expand=True, fill="both")
btn_load = tk.Button(frame_left, text="Tải ảnh gốc", command=load_image)
btn_load.pack(pady=5)

# ---- Khung giữa: Nút dự đoán ----
frame_center = tk.Frame(frame_main)
frame_center.pack(side="left", padx=50)
btn_predict = tk.Button(frame_center, text="Dự đoán", command=predict_animal)
btn_predict.pack(pady=10)

# ---- Khung phải: Ảnh kết quả ----
frame_right = tk.Frame(frame_main)
frame_right.pack(side="left", padx=30)
box_right = tk.Frame(frame_right, bd=2, relief="groove", width=IMG_WIDTH, height=IMG_HEIGHT)
box_right.pack()
box_right.pack_propagate(False)
lbl_image_right = tk.Label(box_right, text="Ảnh Kết quả")
lbl_image_right.pack(expand=True, fill="both")
lbl_result_text = tk.Label(frame_right, text="Ảnh Kết quả chuyển đổi")
lbl_result_text.pack(pady=5)



# Khởi chạy vòng lặp chính của giao diện
root.mainloop()