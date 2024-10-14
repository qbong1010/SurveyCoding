import os
import cv2
import logging
from pdf2image import convert_from_path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_directory_exists(directory):
    """지정된 디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"디렉토리 생성됨: {directory}")
    else:
        logging.info(f"디렉토리 존재: {directory}")

def generate_image_filename(output_folder, page_number, extension="jpeg"):
    """페이지 번호에 기반하여 파일명 생성"""
    return os.path.join(output_folder, f"page_{page_number}.{extension}")

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    """PDF 파일을 페이지별로 이미지로 변환 후 저장"""
    logging.info(f"PDF 변환 시작: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    for i, image in enumerate(images, start=1):
        image_path = generate_image_filename(output_folder, i)
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
        logging.info(f"페이지 {i} 저장 완료: {image_path}")

    return image_paths

def process_image(image_path):
    """이미지 대비 조정 및 흑백 변환 후 저장"""
    logging.info(f"이미지 처리 시작: {image_path}")
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"이미지를 읽는 데 실패했습니다: {image_path}")
        return

    # 이미지를 흑백으로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 명도 대비 조정 (CLAHE 사용)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    contrast_image = clahe.apply(gray_image)
    
    # 처리된 이미지 저장 (흑백 및 대비 보정된 이미지)
    cv2.imwrite(image_path, contrast_image)
    logging.info(f"이미지 처리 및 저장 완료: {image_path}")

def process_pdf_images(pdf_path, output_folder, dpi=300):
    """PDF 파일을 이미지로 변환하고 처리"""
    ensure_directory_exists(output_folder)
    
    # PDF 파일을 이미지로 변환
    image_paths = convert_pdf_to_images(pdf_path, output_folder, dpi=dpi)
    
    # 각 이미지에 대해 흑백 및 대비 보정 처리
    for image_path in image_paths:
        process_image(image_path)

# 사용 예시
if __name__ == "__main__":
    pdf_path = r"C:\Users\dohwa\Desktop\MyPlugin\SurveyCoding\Completed\진건읍_test.pdf"
    output_folder = r"C:\Users\dohwa\Desktop\MyPlugin\SurveyCoding\Preprocessed\output_images"
    
    process_pdf_images(pdf_path, output_folder)
