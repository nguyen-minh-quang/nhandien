import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from translate import translate
import matplotlib.pyplot as plt

# Kích thước ảnh đầu vào cho mô hình
IMG_SIZE = (150, 150)
BASE_DIR = "dataset/raw-img/"


def load_and_preprocess_data():
    images = []
    labels = []
    class_names_raw = sorted(os.listdir(BASE_DIR))

    # Lọc bỏ các file không phải thư mục
    class_names_raw = [d for d in class_names_raw if os.path.isdir(os.path.join(BASE_DIR, d))]

    # Tạo danh sách tên lớp bằng tiếng Anh
    class_names = [translate[name] for name in class_names_raw]

    print("Đang tải và tiền xử lý dữ liệu...")

    for i, class_name_raw in enumerate(class_names_raw):
        class_path = os.path.join(BASE_DIR, class_name_raw)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    # Chuyển đổi màu sắc (mô hình thường dùng RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize ảnh về cùng kích thước
                    img = cv2.resize(img, IMG_SIZE)

                    images.append(img)
                    labels.append(i)  # Lưu index của class
                except Exception as e:
                    print(f"Không thể đọc ảnh {img_path}: {e}")

    images = np.array(images, dtype="float32") / 255.0  # Chuẩn hóa về [0, 1]
    labels = np.array(labels)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Chuyển đổi nhãn sang one-hot encoding
    num_classes = len(class_names)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"Số lượng ảnh huấn luyện: {len(X_train)}")
    print(f"Số lượng ảnh kiểm tra: {len(X_test)}")

    return X_train, X_test, y_train, y_test, class_names


def build_model(num_classes):
    model = Sequential([
        # Lớp tích chập đầu tiên
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D((2, 2)),

        # Lớp tích chập thứ hai
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Lớp tích chập thứ ba
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Lớp tích chập thứ tư
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Làm phẳng dữ liệu đầu ra từ lớp tích chập
        Flatten(),

        # Lớp Dense (fully connected)
        Dense(512, activation='relu'),
        Dropout(0.5),  # Tăng khả năng tổng quát hóa, giảm overfitting

        # Lớp đầu ra với số neuron bằng số lớp động vật
        Dense(num_classes, activation='softmax')
    ])

    # Biên dịch mô hình
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data()
    num_classes = len(class_names)

    model = build_model(num_classes)
    model.summary()

    # Định nghĩa các callbacks
    callbacks = [
        # Lưu mô hình tốt nhất dựa trên độ chính xác kiểm tra
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        # Dừng sớm nếu không có sự cải thiện sau 10 epochs
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        # Giảm learning rate khi mô hình không cải thiện
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
    ]

    # Bắt đầu huấn luyện mô hình với Data Augmentation
    print("\nBắt đầu huấn luyện mô hình với Data Augmentation...")

    # Tạo một trình tạo dữ liệu để tăng cường ảnh huấn luyện
    datagen = ImageDataGenerator(
        rotation_range=20,  # Xoay ảnh ngẫu nhiên 20 độ
        width_shift_range=0.2,  # Dịch chuyển chiều rộng ảnh ngẫu nhiên 20%
        height_shift_range=0.2,  # Dịch chuyển chiều cao ảnh ngẫu nhiên 20%
        shear_range=0.2,  # Biến đổi xiên ảnh
        zoom_range=0.2,  # Phóng to/thu nhỏ ảnh ngẫu nhiên
        horizontal_flip=True,  # Lật ảnh theo chiều ngang
        fill_mode='nearest'  # Điền vào các pixel mới bằng pixel gần nhất
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,  # Số epochs có thể điều chỉnh
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1  # Hiển thị tiến độ trong terminal
    )

    print("\nĐánh giá mô hình trên tập kiểm tra...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Độ chính xác trên tập kiểm tra: {accuracy:.4f}")

    # Hiển thị biểu đồ độ chính xác và mất mát
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Độ chính xác huấn luyện')
    plt.plot(history.history['val_accuracy'], label='Độ chính xác kiểm tra')
    plt.title('Độ chính xác của mô hình')
    plt.xlabel('Epoch')
    plt.ylabel('Độ chính xác')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Mất mát huấn luyện')
    plt.plot(history.history['val_loss'], label='Mất mát kiểm tra')
    plt.title('Mất mát của mô hình')
    plt.xlabel('Epoch')
    plt.ylabel('Mất mát')
    plt.legend()

    plt.show()