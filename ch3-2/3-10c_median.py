import cv2
import numpy as np

#원본 이미지
gray=cv2.imread('lenna256.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Smooth', gray)

#medianBlur 필터
median = cv2.medianBlur(gray, 5) #blur 필터는 5,5 로 적었어야 했는데, medianBlur필터는 5라고만 적으면 됨
cv2.imshow('Smooth - Median',median)

#Blur 필터
blur = cv2.blur(gray, (5,5))
cv2.imshow('Smooth - blur', blur)


#1 다양한 크기의 필터
median1=np.hstack((gray,cv2.medianBlur(gray, 3),cv2.medianBlur(gray, 7),cv2.medianBlur(gray, 11)))
cv2.imshow('Smooth - Median1',median1)

cv2.waitKey()
cv2.destroyAllWindows()