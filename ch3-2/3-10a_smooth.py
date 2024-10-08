import cv2
import numpy as np

gray=cv2.imread('lenna256.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original',gray)

# 1 blur
faverage=np.array([[1.0/9.0, 1.0/9.0, 1.0/9.0],
                   [ 1.0/9.0, 1.0/9.0, 1.0/9.0],
                   [ 1.0/9.0, 1.0/9.0, 1.0/9.0]])
average1 = cv2.filter2D(gray, -1, faverage) #gray = 입력 영상, faverage = 출력 영상 =>필터로 블러 효과 내는거임

average2 = cv2.blur(gray,(3,3)) #필터 크기 커지면 다 적기 너무 힘드니까 blur 사용해서 하는거임
cv2.imshow('Average - filter2D',average1)
cv2.imshow('Average - blur',average2)


#2 다양한 크기의 스무딩 필터, 블러링 마스크 크기 커질 수록 더 많이 블러링 됨
smooth=np.hstack((gray,cv2.blur(gray,(3,3),0.0),cv2.blur(gray,(7,7),0.0),cv2.blur(gray,(11,11),0.0),cv2.blur(gray,(15,15),0.0)))
cv2.imshow('Smooth',smooth)

#3 Gaussian -> blur보다는 더 선명해 보임
blur5 = cv2.blur(gray, (5,5))
gaussian = cv2.GaussianBlur(gray,(5,5),1.0)
gaussian1=np.hstack((gray,blur5, gaussian))
cv2.imshow('Smooth - gaussian1',gaussian1)

#4 Gaussian - 다양한 크기의 필터, 마스크 크기 커질 수록 블러링 더 되긴하는데, blur보다는 차이 많지 않음
gaussian2=np.hstack((gray,cv2.GaussianBlur(gray,(3,3),1.0),cv2.GaussianBlur(gray,(7,7),1.0),cv2.GaussianBlur(gray,(11,11),1.0),cv2.GaussianBlur(gray,(15,15),1.0)))
cv2.imshow('Smooth - Gaussian2',gaussian2)

#5 Gaussian - sigma는 보통 0, 값이 클수록 블러링 효과 커짐, 시그마 값 커질수록 더 blur많이 들어감
gaussian3=np.hstack((gray,cv2.GaussianBlur(gray,(5,5),1.0),cv2.GaussianBlur(gray,(5,5),3.0),cv2.GaussianBlur(gray,(5,5),7.0),cv2.GaussianBlur(gray,(5,5),11.0)))
#cv2.imshow('Smooth - Gaussian3',gaussian3)

cv2.waitKey()
cv2.destroyAllWindows()