import cv2
import numpy as np

color=cv2.imread('lenna256.png')

# Special effects
bila = cv2.bilateralFilter(color, -1,10,5)

edgep=cv2.edgePreservingFilter(color, flags=1, sigma_s=60, sigma_r=0.4)

sty=cv2.stylization(color,sigma_s=60, sigma_r=0.45)

graySketch, colorSketch = cv2.pencilSketch(color, sigma_s=60, sigma_r=0.7, shade_factor=0.02)

oil=cv2.xphoto.oilPainting(color, 7, 1) #xphoto 최근에 추가된 라이브러리라서 설치해야 함 

cgraySketch=cv2.cvtColor(graySketch,cv2.COLOR_GRAY2BGR) #이미지 다 연결하려고 hstack으로 묶기위해 그레이이미지도 칼라이미지(3바이트)도 바꿔준거임
special=np.hstack((color, bila, edgep, sty, cgraySketch, colorSketch, oil))
cv2.imshow('Special Effects',special)

cv2.waitKey()
cv2.destroyAllWindows()