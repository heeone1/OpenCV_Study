import cv2
import numpy as np

img=cv2.imread('rose.png')
cv2.imshow('Original',img)

img=cv2.rectangle(img,(180,290),(220,330),(255,0,0),2)
patch=img[290:330,180:220,:] #사진의 일부부만 patch라는 변수에 저장

# patch0 = cv2.resize(patch, (480, 480))
#모두 가로 세로로 12배 확대
patch1=cv2.resize(patch,dsize=(0,0),fx=12,fy=12,interpolation=cv2.INTER_NEAREST) # 화질이 별로
patch2=cv2.resize(patch,dsize=(0,0),fx=12,fy=12,interpolation=cv2.INTER_LINEAR) # INTER_NEAREST보다 화질 좋음
patch3=cv2.resize(patch,dsize=(0,0),fx=12,fy=12,interpolation=cv2.INTER_CUBIC) # INTER_NEAREST보다 화질 좋음

dst1=np.hstack((patch1,patch2,patch3))
cv2.imshow('Resize - zoomin',dst1)

# img_small0 = cv2.resize(img, dsize=(297, 198))
img_small1 = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)  # 1/4로 축소
img_small2 = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)  # 1/4로 축소, 기본값 interpolation=cv2.INTER_LINEAR
img_small3 = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)  # 1/4로 축소

dst2=np.hstack((img_small1,img_small2,img_small3))
cv2.imshow('Resize - zoomout',dst2)

cv2.waitKey()
cv2.destroyAllWindows()