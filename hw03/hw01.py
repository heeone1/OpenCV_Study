import cv2

cap = cv2.VideoCapture("hand.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   # 비디오 크기 지정
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'XVID')        # 비디오 저장 방식 지정
outV = cv2.VideoWriter('./record.mp4', fourcc, fps, frame_size) # 비디오 저장 객체 생성

while True:  # 무한루프로
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득(frame)
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    #BGR에서 HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 피부색 범위 설정
    lower_skin = (0, 30, 0)
    upper_skin = (20, 180, 255)

    # 피부색 영역 검출
    dst1 = cv2.inRange(hsv, lower_skin, upper_skin)

    # 모폴로지 연산으로 잡음 제거
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dst1 = cv2.dilate(dst1, se2, iterations=1)

    # 윤곽선 검출
    contours, _ = cv2.findContours(dst1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        # 가장 큰 윤곽선
        max_contour = max(contours, key=cv2.contourArea)

        # 손 윤곽선을 파란색으로 그리기
        cv2.drawContours(frame, [max_contour], -1, (255, 0, 0), 2)

        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)

        finger_count = 0  # 손가락 수 초기화

        if defects is not None:  # 결함이 있는지 확인
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]

                if d > 8000:  # 깊은 결함을 기준으로 손가락 수 증가
                    finger_count += 1

            # 가위, 바위, 보 판별
            hand = ""
            if finger_count < 1:  # 주먹
                hand = "Rock"
            elif finger_count <= 3:  # 가위
                hand = "Scissors"
            elif finger_count >= 5:  # 보
                hand = "Paper"

            cv2.putText(frame, hand, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    outV.write(frame)  # 비디오로 프레임 저장
    cv2.imshow("Hand Contour", frame)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
