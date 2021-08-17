import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"D:\TUGAS KULIAH\tesseract\tesseract.exe"

img = cv2.imread("ss_3.png")


img = cv2.resize(img, None, fx=0.5, fy=0.5)

# Mengkonversi warna gambar menjadi grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Mengkonversi warna gambar menjadi hitam-putih (menggunakan adaptive threshold)
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

config = "--psm 3"
text = pytesseract.image_to_string(adaptive_threshold, config=config, lang="eng")
print(text)

cv2.imshow("before", gray)
cv2.imshow("after", adaptive_threshold)
cv2.waitKey(0)
