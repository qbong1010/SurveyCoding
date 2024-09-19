import cv2
import numpy as np

drawing = False
ix, iy = -1, -1
boxes = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        img_copy = img.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        boxes.append((ix, iy, x, y))

# 이미지 파일 경로
image_path = 'C:\\Users\\dohwa\\Desktop\\MyPlugin\\SurveyCoding\\Origin\\SurveySheetOrigin_Page2.jpg'

try:
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("이미지를 로드할 수 없습니다.")
    img_copy = img.copy()

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_rectangle)

    print("이미지 창에서 드래그하여 박스를 그리세요. 완료 후 'q'를 눌러 창을 닫으세요.")

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