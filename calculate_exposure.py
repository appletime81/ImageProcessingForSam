import cv2
import numpy as np


def calculate_exposure(image_path):
    # 讀取影像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image not found or unable to read")
        return

    # 計算影像的直方圖
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # 计算亮度的分布
    total_pixels = image.size
    dark_ratio = np.sum(hist[:50]) / total_pixels  # 暗像素的比例
    bright_ratio = np.sum(hist[205:]) / total_pixels  # 亮像素的比例

    # 評估曝光度
    print(f"Dark pixel ratio: {dark_ratio}")
    print(f"Bright pixel ratio: {bright_ratio}")
    if dark_ratio > 0.5:
        return "Underexposed"
    elif bright_ratio > 0.5:
        return "Overexposed"
    else:
        return "Well-exposed"


# 使用示例
image_path = "Image/Image/Golden/10013993PTRS1/Stream2-Golden.jpg"  # 替換為您的圖片路徑
exposure = calculate_exposure(image_path)
print(f"Image Exposure: {exposure}")
