import cv2
import sys
import numpy as np
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환하여 저장
    blur = cv2.blur(gray, (5, 5))
    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, 3)
    _, b_img = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 17), np.uint8)
    morph = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    return morph

def find_candidates(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.minAreaRect(c) for c in contours]  # 외곽 최소 영역
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if verify_aspect_size(size)]

    return candidates

def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h/ w if h > w else w/ h       # 종횡비 계산

    chk1 = 3000 < (h * w) < 12000  # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 6.5      # 번호판 종횡비 조건

    #print(w,h)
    return (chk1 and chk2)


def warp_plate(image, rect):
    center, (w, h), angle = rect  # rect는 중심점, 크기, 회전 각도으로 표시
    w = w + 10
    h = h + 10
    if w < h:  # 세로가 긴 영역이면
        w, h = h, w  # 가로와 세로 맞바꿈
        angle -= 90  # 회전 각도 조정

    size = image.shape[1::-1]  # 행태와 크기는 역순
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)  # 회전 행렬 계산
    rot_img = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)  # 회전 변환

    crop_img = cv2.getRectSubPix(rot_img, (w, h), center)  # 후보영역 가져오기
    crop_img = 255 - cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, warped_bin = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.resize(warped_bin, (288, 56))  # pytesseract로 인식하기 적합한 크기로

def enhance_plate(warped):

    # Threshold로 이진화
    _, warped_bin = cv2.threshold(warped, 60, 255, cv2.THRESH_BINARY)

    # 왼쪽에 검정 사각형 추가
    h, w = warped_bin.shape
    result = warped_bin.copy()

    # 검정 사각형 그리기
    rect_width = int(w * 0.13)  # 왼쪽에 추가할 검정 사각형의 너비
    result[:, :rect_width] = 0  # 이미지의 왼쪽 rect_width만큼 검정색으로 채우기

    return result

def ocr_plate(warped_plate):
    # 번호판 이미지를 OCR로 인식

    text = pytesseract.image_to_string(warped_plate, lang='kor', config='--psm 7')
    text = ''.join(filter(str.isalnum, text))  # 문자 및 숫자만 필터링
    # 첫 번째 문자가 숫자가 아닐 경우, 뒤에서부터 출력
    if text and not text[0].isdigit():
        text = text[1:]  # 첫 번째 문자 제거

    # 숫자가 포함된 경우만 반환
    if any(char.isdigit() for char in text):
        return text.strip()
    else:
        return None


car_no = str(input("자동차 영상 번호 (00~09): "))
img = cv2.imread('cars/'+car_no+'.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
cv2.imshow('original',img)

# 1 전처리 단계 (hw2-2)
preprocessed = preprocessing(img)

cv2.imshow('plate candidate 0',preprocessed)

# 2 번호판 후보 영역 검출 (hw3-2)
candidates = find_candidates(preprocessed)

if not candidates:
    print("번호판 후보를 찾을 수 없습니다.")
else:
    img2 = img.copy()
    for i, candidate in enumerate(candidates):
        pts = np.int32(cv2.boxPoints(candidate))
        cv2.polylines(img2, [pts], True, (0, 225, 255), 3)

        # 번호판 영역
        warped = warp_plate(img, candidate)

        # 이미지 품질 개선
        enhanced_warped = enhance_plate(warped)

        # OCR 처리
        img_pil = Image.fromarray(enhanced_warped)
        recognized_text = ocr_plate(img_pil)
        if recognized_text:
            print(recognized_text)
            cv2.imshow(f'Warped Plate {i}', warped)

    cv2.imshow('Plate Candidates', img2)

cv2.waitKey()
cv2.destroyAllWindows()