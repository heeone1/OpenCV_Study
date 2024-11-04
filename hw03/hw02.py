import cv2
import numpy as np

def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h/ w if h > w else w/ h # 종횡비 계산

    chk1 = 3000 < (h * w) < 12000 # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 6.5 # 번호판 종횡비 조건

    return (chk1 and chk2)

car_no = input("자동차 영상 번호 (00 to 05): ")

color_image = cv2.imread('cars/' +car_no+ '.jpg')
gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# 잡음 제거
blur = cv2.blur(gray, (3, 3))

# 수직 에지
prewitt_filter_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_grad_x = cv2.filter2D(blur, -1, prewitt_filter_x)

# 검은 배경과 흰 에지 분리
_, th = cv2.threshold(prewitt_grad_x, 145, 255, cv2.THRESH_BINARY)

# close
se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 5))
b_closing = cv2.erode(cv2.dilate(th, se2, iterations=3), se2, iterations=3)

#윤곽선 찾기
contours, _ = cv2.findContours(b_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
candidates = []

car3 = color_image.copy()  # 모든 윤곽선을 그릴 이미지
car4 = color_image.copy()  # 유효한 윤곽선만 그릴 이미지

for cnt in contours:
    # 최소면적 사각형 찾기
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.drawContours(car3, [box], 0, (0, 255, 255), 2)

    # 가로 세로 비율 등의 조건을 통해 자동차 번호판 후보 필터링
    width, height = rect[1]
    if verify_aspect_size((width, height)):
        candidates.append(box)
        # 번호판 후보를 이미지에 표시
        cv2.drawContours(car4, [box], 0, (0, 255, 0), 2)

b_closing = cv2.cvtColor(b_closing, cv2.COLOR_GRAY2BGR)

# 각 이미지의 크기 줄이기
color_image_resized = cv2.resize(color_image, (color_image.shape[1] // 2, color_image.shape[0] // 2))
b_closing_resized = cv2.resize(b_closing, (color_image.shape[1] // 2, color_image.shape[0] // 2))
car3_resized = cv2.resize(car3, (color_image.shape[1] // 2, color_image.shape[0] // 2))
car4_resized = cv2.resize(car4, (color_image.shape[1] // 2, color_image.shape[0] // 2))

# 이미지 연결
morphology = np.hstack((color_image_resized, b_closing_resized, car3_resized, car4_resized))

# 결과 출력
cv2.imshow('Morphology', morphology)
cv2.waitKey()
cv2.destroyAllWindows()
