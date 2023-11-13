import cv2
import numpy as np
import os


def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def nothing(x):
    pass


cv2.namedWindow("demo", 0)  # 將顯示視窗的大小適應於顯示器的分辨率
cv2.createTrackbar("Value of Gamma", "demo", 100, 1000, nothing)  # 使用滑動條動態調整參數gamma

data_base_dir = "./Image/Image/Input/10013993PTRS1/Testing_1/"  # 輸入資料夾的路徑
outfile_dir = "./"  # 輸出資料夾的路徑
processed_number = 0  # 統計處理圖片的數量
print("press enter to make sure your operation and process the next picture")

for file in os.listdir(data_base_dir):  # 遍歷目標資料夾圖片
    read_img_name = os.path.join(data_base_dir, file)  # 讀取圖片路徑
    image = cv2.imread(read_img_name)  # 讀入圖片

    while 1:
        value_of_gamma = cv2.getTrackbarPos("Value of Gamma", "demo")  # gamma取值
        value_of_gamma = value_of_gamma * 0.01  # 壓縮gamma範圍，以進行精細調整
        image_gamma_correct = gamma_trans(
            image, value_of_gamma
        )  # 2.5為gamma函數的指數值，大於1曝光度下降，大於0小於1曝光度增強
        cv2.imshow("demo", image_gamma_correct)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:  # 按下enter鍵確認處理、儲存圖片到輸出資料夾和讀取下一張圖片
            processed_number += 1
            out_img_name = os.path.join(outfile_dir, file)
            cv2.imwrite(out_img_name, image_gamma_correct)
            print("The number of photos which were processed is ", processed_number)
            break
