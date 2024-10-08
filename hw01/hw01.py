import cv2
import numpy as np

# 600*900
canvas = np.ones((600,900,3), np.uint8) * 255

BrushSize = 5 # 붓의 크기
RColor = (0,0,255) # 빨간색
GColor = (0, 255, 0) # 초록색
BColor = (255,0,0) # 파란색
YColor = (0,255,255) # 노란색
MColor = (255,255,0) # 민트색
PColor = (255,0,255) # 핑크색

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def draw(event, x, y, flags, param):
    global x1, y1

    if event==cv2.EVENT_LBUTTONDOWN:
        x1,y1 = x, y

    elif event==cv2.EVENT_RBUTTONDOWN:
        x1,y1 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        d = int(distance(x1, y1, x, y))
        if flags & cv2.EVENT_FLAG_ALTKEY:
            cv2.rectangle(canvas, (x1, y1), (x, y), PColor, 2)  # 직사각형
        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            cv2.circle(canvas, (x, y), d, MColor, 2) # 원

    elif event == cv2.EVENT_RBUTTONUP:
        d = int(distance(x1, y1, x, y))
        if flags & cv2.EVENT_FLAG_ALTKEY:
            cv2.rectangle(canvas, (x1, y1), (x, y), PColor, -1)  # 칠해진 직사각형
        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            cv2.circle(canvas, (x, y), d, MColor, -1) # 칠해진 원

    elif event == cv2.EVENT_MOUSEMOVE and flags&cv2.EVENT_FLAG_LBUTTON:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                cv2.circle(canvas, (x, y), BrushSize, GColor, -1)
            elif not (flags & cv2.EVENT_FLAG_ALTKEY or flags & cv2.EVENT_FLAG_CTRLKEY):
                cv2.circle(canvas, (x, y), BrushSize, BColor, -1)

    elif event == cv2.EVENT_MOUSEMOVE and flags&cv2.EVENT_FLAG_RBUTTON:
            if flags&cv2.EVENT_FLAG_SHIFTKEY:
                cv2.circle(canvas, (x,y), BrushSize, YColor, -1)
            elif not (flags & cv2.EVENT_FLAG_ALTKEY or flags & cv2.EVENT_FLAG_CTRLKEY):
                cv2.circle(canvas, (x, y), BrushSize, RColor, -1)

    cv2.imshow('Painting', canvas)  # 수정된 이미지를 다시 그림

cv2.namedWindow('Painting')
cv2.imshow('Painting',canvas)

cv2.setMouseCallback('Painting',draw)

while(True):
    if cv2.waitKey(1)==ord('s'):
        cv2.imwrite('painting.png', canvas) # 이미지 저장
    elif cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows() # 모든 창 닫기
        break