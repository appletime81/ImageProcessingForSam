import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

img1 = cv2.imread(
    "Image/Input/10013993PTRS1/Testing_2/10013993PTRS1_TWNMLR02V6D00060D1_19_1_20231107102222_neg.png"
)
# img1 = cv2.imread(
#     "Image/Input/10014010PTRU1/Testing_1/10014010PTRU1_NOSTRIPID_1_1_20231109054049.jpg"
# )
img2 = cv2.imread("Image/Golden/10013993PTRS1/Stream1-Golden.jpg")

# ------------------------------- convert to gray --------------------------------
img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# --------------------------------------------------------------------------------

# ------------------------------------ binary ------------------------------------
output2 = cv2.adaptiveThreshold(
    img_gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 2
)
output3 = cv2.adaptiveThreshold(
    img_gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 2
)
# --------------------------------------------------------------------------------

# --------------------------------- filter noise ---------------------------------
median1 = cv2.medianBlur(output2, 13)
median2 = cv2.medianBlur(output3, 13)
# --------------------------------------------------------------------------------

# ----------------------------------- cal SSIM -----------------------------------
ssim_med_1vs2 = ssim(median1, median2, data_range=median2.max() - median2.min())
print(ssim_med_1vs2)
# --------------------------------------------------------------------------------

# cv2.imshow('ret002', output2)
# cv2.imshow('ret003', output3)
# cv2.imshow('ret_dst', dst)
cv2.imshow("ret_median1", median1)
cv2.imshow("ret_median2", median2)
cv2.waitKey(0)
cv2.destroyAllWindows()
