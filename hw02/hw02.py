import cv2
import numpy as np

def car (img):
    gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # 잡음 제거
    blur = cv2.blur(gray, (3, 3))

    # 수직 에지
    prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_grad_x = cv2.filter2D(blur, -1, prewitt_filter_x)

    # 검은 배경과 흰 에지 분리
    _, th = cv2.threshold(prewitt_grad_x, 100, 255, cv2.THRESH_BINARY)

    # close
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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