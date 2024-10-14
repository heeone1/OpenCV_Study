import cv2
import numpy as np

def car (img):
    color_image = cv2.imread(img)

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 잡음 제거
    blur = cv2.blur(gray, (3, 3))

    # 수직 에지
    prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_grad_x = cv2.filter2D(blur, -1, prewitt_filter_x)

    # 검은 배경과 흰 에지 분리
    _, th = cv2.threshold(prewitt_grad_x, 150, 255, cv2.THRESH_BINARY)

    # close
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 5))
    b_closing = cv2.erode(cv2.dilate(th, se2, iterations=3), se2, iterations=3)

    prewitt_grad_x = cv2.cvtColor(prewitt_grad_x, cv2.COLOR_GRAY2BGR)
    th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    b_closing = cv2.cvtColor(b_closing, cv2.COLOR_GRAY2BGR)

    # 각 이미지의 크기 줄이기
    color_image_resized = cv2.resize(color_image, (color_image.shape[1]//2, color_image.shape[0]//2))
    prewitt_grad_x_resized = cv2.resize(prewitt_grad_x, (color_image.shape[1]//2, color_image.shape[0]//2))
    th_resized = cv2.resize(th, (color_image.shape[1]//2, color_image.shape[0]//2))
    b_closing_resized = cv2.resize(b_closing, (color_image.shape[1]//2, color_image.shape[0]//2))

    # 이미지 연결
    morphology = np.hstack((color_image_resized, prewitt_grad_x_resized, th_resized, b_closing_resized))

    # 결과 출력
    cv2.imshow('Morphology', morphology)

    cv2.waitKey()
    cv2.destroyAllWindows()

car ('cars/00.jpg')
car ('cars/01.jpg')
car ('cars/02.jpg')
car ('cars/03.jpg')
car ('cars/04.jpg')
car ('cars/05.jpg')