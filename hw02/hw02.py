import cv2
import numpy as np

def car (img):
    gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    blur = cv2.blur(gray, (3, 3))

    prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_grad_x = cv2.filter2D(blur, -1, prewitt_filter_x)

    _, th = cv2.threshold(prewitt_grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Sobel 에지 결과에 임계값 구해서 이진화

    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 5))
    b_closing = cv2.erode(cv2.dilate(th, se2, iterations=3), se2, iterations=3)

    cv2.imshow('Morphology', b_closing)

    cv2.waitKey()
    cv2.destroyAllWindows()

car ('cars/00.jpg')
car ('cars/01.jpg')
car ('cars/02.jpg')
car ('cars/03.jpg')
car ('cars/04.jpg')
car ('cars/05.jpg')






# gray0=cv2.imread('cars/00.jpg', cv2.IMREAD_GRAYSCALE)
# gray1=cv2.imread('cars/01.jpg', cv2.IMREAD_GRAYSCALE)
# gray2=cv2.imread('cars/02.jpg', cv2.IMREAD_GRAYSCALE)
# gray3=cv2.imread('cars/03.jpg', cv2.IMREAD_GRAYSCALE)
# gray4=cv2.imread('cars/04.jpg', cv2.IMREAD_GRAYSCALE)
# gray5=cv2.imread('cars/05.jpg', cv2.IMREAD_GRAYSCALE)
#
# # 잡음 제거
# blur0=cv2.blur(gray0,(3,3))
# blur1=cv2.blur(gray1,(3,3))
# blur2=cv2.blur(gray2,(3,3))
# blur3=cv2.blur(gray3,(3,3))
# blur4=cv2.blur(gray4,(3,3))
# blur5=cv2.blur(gray5,(3,3))
#
# # 수직 에지
# prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
# prewitt_grad_x = cv2.filter2D(blur0, -1, prewitt_filter_x)
#
# # 검은 배경과 흰 에지 분리
# # _, th = cv2.threshold(prewitt_grad_x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Sobel 에지 결과에 임계값 구해서 이진화
# _, th = cv2.threshold(prewitt_grad_x, 100, 255, cv2.THRESH_BINARY)
#
# # close
# se2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 5))
# b_closing=cv2.erode(cv2.dilate(th,se2,iterations=3),se2,iterations=3)
#
#
# morphology=np.hstack((prewitt_grad_x, th, b_closing))
# cv2.imshow('Morphology',morphology)
#
# cv2.waitKey()
# cv2.destroyAllWindows()