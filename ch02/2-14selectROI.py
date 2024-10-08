import cv2
import sys

img = cv2.imread('girl_laughing.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

x, y, w, h = cv2.selectROI(img) 	# 관심 영역을 선택후 space 또는 enter
print("Selected ROI:", x, y, w, h)

roi = img[y:y+h, x:x+w]	# 선택한 영역만 잘라냄

#mosaic - 3-2 smoothing
# blur_roi = cv2.blur(roi, (25,25))

#mosaic - 3-3 resize
resize_roi = cv2.resize(roi, (w//15, h//15))
resize_roi = cv2.resize(resize_roi, (w, h), interpolation=cv2.INTER_NEAREST) #모자이크

img[y:y+h, x:x+w] = resize_roi

#cv2.imwrite('roi.jpg', roi)	# 잘라낸 영역을 저장

#cv2.imshow('ROI', roi)
cv2.imshow('ROI-Blur', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
