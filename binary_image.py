import cv2
import numpy as np

img1 = cv2.imread('Image/Input/10013993PTRS1/Testing_2/10013993PTRS1_TWNMLR02V6D00060D1_19_3_20231107102222.jpg')
img2 = cv2.imread('Image/Golden/10013993PTRS1/Stream3-Golden.jpg')
img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY); # 轉換前，都先將圖片轉換成灰階色彩
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY); # 轉換前，都先將圖片轉換成灰階色彩
print(img_gray2.shape)
ret1, output1 = cv2.threshold(img_gray1, 127, 255, cv2.THRESH_BINARY)
ret2, output2 = cv2.threshold(img_gray2, 127, 255, cv2.THRESH_BINARY)
output2 = cv2.adaptiveThreshold(img_gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 2)
output3 = cv2.adaptiveThreshold(img_gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
dst = cv2.fastNlMeansDenoising(output3, None, 10, 15, 75)


# cv2.imshow('oxxostudio', img)
# cv2.imshow('oxxostudio1', output1)
cv2.imshow('oxxostudio2', output2)
cv2.imshow('oxxostudio3', output3)
cv2.imshow('oxxostudio4', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
