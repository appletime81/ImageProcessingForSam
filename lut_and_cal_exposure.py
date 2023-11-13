import cv2
import numpy as np
import os


def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def calculate_exposure(img):
    if img is None:
        print("Image not found or unable to read")
        return

    # 計算影像的直方圖
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 計算亮度的分佈
    total_pixels = img.size
    dark_ratio = np.sum(hist[:50]) / total_pixels  # 暗像素的比例
    bright_ratio = np.sum(hist[205:]) / total_pixels  # 亮像素的比例

    # 評估曝光度
    print(f"Dark pixel ratio: {dark_ratio}")
    print(f"Bright pixel ratio: {bright_ratio}")
    return dark_ratio, bright_ratio


def modify_dark_ratio_and_bright_ratio(img_path):
    img = cv2.imread(img_path)

    for i in range(1, 10000, 1):
        value_of_gamma = i * 0.01
        print(i)
        image_gamma_correct = gamma_trans(img, value_of_gamma)
        cv2.imwrite("temp_image_gamma_correct.jpg", image_gamma_correct)

        gray_scale_image_gamma_correct = cv2.imread(
            "temp_image_gamma_correct.jpg", cv2.IMREAD_GRAYSCALE
        )

        print(f" Result {i} ".center(50, "-"))
        dark_ratio, bright_ratio = calculate_exposure(gray_scale_image_gamma_correct)

        # if dark_ratio >= 0.556142578125 and dark_ratio <= 0.029127604166666668:
        if dark_ratio >= 0.6:
            # if dark_ratio >= 0.556142578125:
            # save image_gamma_correct
            print("save image_gamma_correct")
            cv2.imwrite(
                "test.jpg",
                image_gamma_correct,
            )
            break
        elif (
            dark_ratio >= 0.556142578125
            and dark_ratio >= 0.029127604166666668
            and i == 9999
        ):
            print("save image_gamma_correct")
            cv2.imwrite(
                "test.jpg",
                image_gamma_correct,
            )
            break
    os.system("del temp_image_gamma_correct.jpg")


def median_filter(img_path):
    img = cv2.imread(img_path)
    median = cv2.medianBlur(img, 3)
    cv2.imwrite("test_after_median.jpg", median)
    return median


def fast_nlmeans_denoising_colored(img_path):
    img = cv2.imread(img_path)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imwrite("test_after_fast_nlmeans_denoising_colored.jpg", dst)
    return dst


if __name__ == "__main__":
    # img_path = "Image/Input/10013993PTRS1/Testing_1/10013993PTRS1_NOSTRIPID_4_2_20231107214203.jpg"
    img_path = "darkened_image.jpg"
    modify_dark_ratio_and_bright_ratio(img_path)
    # median_filter(img_path="test.jpg")
    _ = fast_nlmeans_denoising_colored(img_path="test.jpg")
