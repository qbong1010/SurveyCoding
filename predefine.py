import cv2
import numpy as np

boxes = []

def draw_box(event, x, y, flags, param):
    global img, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭 지점을 중심으로 25x25 박스 좌표 계산
        half_size = 12  # 25의 절반에서 1을 뺀 값
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(img.shape[1] - 1, x + half_size)
        y2 = min(img.shape[0] - 1, y + half_size)

        # 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img_copy = img.copy()

        # 박스 좌표 저장
        boxes.append((x1, y1, x2, y2))

# 이미지 파일 경로
image_path = 'C:\\Users\\dohwa\\Desktop\\MyPlugin\\SurveyCoding\\Origin\\SurveySheetOrigin_Page2.jpg'

try:
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("이미지를 로드할 수 없습니다.")
    img_copy = img.copy()

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_box)

    print("이미지를 클릭하여 25x25 박스를 그리세요. 완료 후 'q'를 눌러 창을 닫으세요.")

    while True:
        cv2.imshow('Image', img_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

    # 추출된 박스 좌표 출력
    print("추출된 박스 좌표:", boxes)

    # 좌표값을 파일로 저장
    with open('box_coordinates.txt', 'w') as f:
        for box in boxes:
            f.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")
    print("박스 좌표값이 box_coordinates.txt 파일에 저장되었습니다.")

except FileNotFoundError as e:
    print(f"오류: {e}")
    exit(1)