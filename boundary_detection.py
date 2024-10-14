import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def find_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)

def approximate_contour(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_page_contour(contours):
    for contour in contours:
        approx = approximate_contour(contour)
        if len(approx) == 4:
            return approx
    return None

def find_content_contour(contours, page_contour, image_shape):
    page_area = cv2.contourArea(page_contour)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < page_area * 0.9 and area > page_area * 0.6:
            approx = approximate_contour(contour)
            if len(approx) == 4:
                return approx
    
    # 컨텐츠 영역을 찾지 못한 경우, 페이지 경계에서 약간 축소된 영역 반환
    page_points = order_points(page_contour.reshape(4, 2))
    margin = min(image_shape) * 0.1
    content_points = np.array([
        page_points[0] + [margin, margin],
        page_points[1] + [-margin, margin],
        page_points[2] + [-margin, -margin],
        page_points[3] + [margin, -margin]
    ])
    return content_points.astype(int)

def detect_boundaries(image_path):
    image = cv2.imread(image_path)
    preprocessed = preprocess_image(image)
    contours = find_contours(preprocessed)
    
    page_contour = find_page_contour(contours)
    if page_contour is None:
        return None, None
    
    content_contour = find_content_contour(contours, page_contour, image.shape[:2])
    
    return order_points(page_contour.reshape(4, 2)), order_points(content_contour.reshape(4, 2))

class ImageViewer:
    def __init__(self, window_name, image, save_path):  # content_boundary 제거
        self.window_name = window_name
        self.original_image = image
        self.displayed_image = image.copy()
        self.save_path = save_path  # save_path 인스턴스 변수로 설정
        
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        
        self.offset_x = 0
        self.offset_y = 0
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        cv2.createTrackbar('Zoom', self.window_name, 0, 100, self.on_zoom)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        self.save_path = save_path  # save_path 인스턴스 변수로 설정
        self.content_boundary = None
        
        self.update_view()

    def on_zoom(self, value):
        self.zoom_factor = self.min_zoom + (value / 100.0) * (self.max_zoom - self.min_zoom)
        self.update_view()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
            else:
                self.zoom_factor = max(self.zoom_factor / 1.1, self.min_zoom)
            zoom_value = int((self.zoom_factor - self.min_zoom) / (self.max_zoom - self.min_zoom) * 100)
            cv2.setTrackbarPos('Zoom', self.window_name, zoom_value)
            self.update_view()
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.offset_x -= (x - self.last_x) / self.zoom_factor
                self.offset_y -= (y - self.last_y) / self.zoom_factor
                self.update_view()
        
        self.last_x = x
        self.last_y = y
    
    def update_view(self):
        height, width = self.original_image.shape[:2]
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        
        zoomed = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        
        start_x = int(max(0, min(new_width - 800, self.offset_x * self.zoom_factor)))
        start_y = int(max(0, min(new_height - 600, self.offset_y * self.zoom_factor)))
        
        end_x = min(start_x + 800, new_width)
        end_y = min(start_y + 600, new_height)
        
        canvas_x = 0
        canvas_y = 0
        
        if start_x < 0:
            canvas_x = -start_x
            start_x = 0
        if start_y < 0:
            canvas_y = -start_y
            start_y = 0
        
        canvas[canvas_y:canvas_y+end_y-start_y, canvas_x:canvas_x+end_x-start_x] = zoomed[start_y:end_y, start_x:end_x]
        
        cv2.imshow(self.window_name, canvas)

    def run(self):
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('s'):  # 's' 키를 눌러 크롭 후 저장
                self.save_cropped_image()
                break
        cv2.destroyAllWindows()

    def save_cropped_image(self):
        if self.content_boundary is not None:  # content_boundary가 설정되어 있는지 확인
            rect = self.content_boundary.astype(int)
            x1, y1 = rect[0]
            x2, y2 = rect[2]
            cropped_image = self.original_image[y1:y2, x1:x2]
            cv2.imwrite(self.save_path, cropped_image)  # 크롭된 이미지 저장
            print(f"크롭된 이미지가 저장되었습니다: {self.save_path}")
        else:
            print("content_boundary가 설정되지 않았습니다.")

# 사용 예
image_path = 'Preprocessed/output_images/page_1.jpeg'
save_path = 'Preprocessed/output_images/cropped_page_1.jpeg'
page_boundary, content_boundary = detect_boundaries(image_path)

if page_boundary is not None and content_boundary is not None:
    image = cv2.imread(image_path)
    cv2.drawContours(image, [page_boundary.astype(int)], 0, (0, 255, 0), 2)
    cv2.drawContours(image, [content_boundary.astype(int)], 0, (0, 0, 255), 2)

    viewer = ImageViewer('Document Boundaries', image, save_path)  # save_path 전달
    viewer.content_boundary = content_boundary  # content_boundary 설정
    viewer.run()
else:
    print("경계를 찾을 수 없습니다.")
