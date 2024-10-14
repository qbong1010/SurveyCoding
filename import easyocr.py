import os
import cv2
import numpy as np
import pandas as pd
import easyocr
import torch

# 각 페이지에 대한 번호 위치를 미리 정의합니다.
# 예: {페이지 번호: [(번호, (x, y)), ...]}
predefined_positions = {
    1: [('①', (100, 200)), ('②', (150, 250)), ('③', (200, 300))],
    2: [('①', (120, 220)), ('②', (170, 270)), ('③', (220, 320))],
    3: [('①', (140, 240)), ('②', (190, 290)), ('③', (240, 340))]
}

def load_image_pairs(origin_folder, completed_folder):
    # 폴더 내의 이미지 파일 목록을 가져옵니다.
    origin_files = sorted([f for f in os.listdir(origin_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    completed_files = sorted([f for f in os.listdir(completed_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # 파일명에서 페이지 번호를 추출하여 매칭합니다.
    origin_dict = {}
    for f in origin_files:
        page_num = extract_page_number(f)
        if page_num is not None:
            origin_dict[page_num] = os.path.join(origin_folder, f)

    completed_dict = {}
    for f in completed_files:
        page_num = extract_page_number(f)
        if page_num is not None:
            completed_dict[page_num] = os.path.join(completed_folder, f)

    # 공통 페이지 번호로 이미지 페어 생성
    common_pages = sorted(set(origin_dict.keys()) & set(completed_dict.keys()))
    image_pairs = []
    for page_num in common_pages:
        origin_img_path = origin_dict[page_num]
        completed_img_path = completed_dict[page_num]
        image_pairs.append((page_num, origin_img_path, completed_img_path))

    return image_pairs

def extract_page_number(filename):
    # 파일명에서 페이지 번호를 추출합니다.
    import re
    match = re.search(r'Page(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def load_images(origin_img_path, completed_img_path):
    origin_img = cv2.imread(origin_img_path)
    completed_img = cv2.imread(completed_img_path)
    if origin_img is None:
        raise FileNotFoundError(f"원본 이미지 파일을 불러올 수 없습니다: {origin_img_path}")
    if completed_img is None:
        raise FileNotFoundError(f"완료된 이미지 파일을 불러올 수 없습니다: {completed_img_path}")
    return origin_img, completed_img

def align_images(origin_img, completed_img):
    # 이미지를 그레이스케일로 변환
    origin_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    completed_gray = cv2.cvtColor(completed_img, cv2.COLOR_BGR2GRAY)

    # 특징점 검출기 및 디스크립터 생성 (ORB 사용)
    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(origin_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(completed_gray, None)

    # 매처 생성 및 특징점 매칭
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # 매칭 결과를 거리 기준으로 정렬
    matches = sorted(matches, key=lambda x: x.distance)

    # 상위 매칭 포인트 사용 (예: 상위 90%)
    good_matches = matches[:int(len(matches) * 0.9)]

    # 매칭된 키포인트 좌표 추출
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # 어피니티 변환 행렬 계산
    if len(src_pts) >= 3:
        matrix, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
        # 완료된 이미지를 원본 이미지에 맞게 변환
        aligned_img = cv2.warpAffine(completed_img, matrix, (origin_img.shape[1], origin_img.shape[0]))
        return aligned_img
    else:
        print("특징점이 충분하지 않아 이미지 정합에 실패했습니다.")
        return completed_img  # 정합 실패 시 원본 이미지 반환

def extract_option_positions(page_num):
    if page_num in predefined_positions:
        return [{'text': text, 'position': position} for text, position in predefined_positions[page_num]]
    else:
        print(f"{page_num}페이지에 대한 사전 정의된 위치가 없습니다.")
        return []

def detect_user_marks(aligned_img, option_positions, page_num):
    # 이미지를 그레이스케일로 변환
    aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

    # 원본 이미지의 번호 위치 주변에 마크가 있는지 확인
    marked_options = []
    for idx, option in enumerate(option_positions):
        x, y = option['position']
        # 번호 주변의 ROI 설정
        roi_size = 20  # 필요에 따라 조정
        x1 = max(x - roi_size, 0)
        y1 = max(y - roi_size, 0)
        x2 = min(x + roi_size, aligned_gray.shape[1])
        y2 = min(y + roi_size, aligned_gray.shape[0])
        roi = aligned_gray[y1:y2, x1:x2]

        # ROI 내의 픽셀 값 분석 (어두운 픽셀이 많으면 마크로 간주)
        _, thresh = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
        black_pixels = cv2.countNonZero(thresh)
        total_pixels = roi.size
        black_ratio = black_pixels / total_pixels

        # 임계값 이상이면 마크가 있다고 판단
        if black_ratio > 0.1:  # 임계값은 이미지에 따라 조정 필요
            marked = True
        else:
            marked = False

        marked_options.append({
            '페이지 번호': page_num,
            '문항 번호': idx + 1,
            '선택한 번호': option['text'] if marked else '선택 안 함'
        })
    return marked_options

def main():
    # 현재 스크립트의 폴더를 기준으로 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    origin_folder = os.path.join(base_dir, 'Origin')
    completed_folder = os.path.join(base_dir, 'Completed')

    # 이미지 페어 로드
    image_pairs = load_image_pairs(origin_folder, completed_folder)

    all_marked_options = []

    for page_num, origin_img_path, completed_img_path in image_pairs:
        print(f"{page_num}페이지 처리 중...")
        try:
            # 이미지 로드
            origin_img, completed_img = load_images(origin_img_path, completed_img_path)

            # 이미지 정합 수행
            aligned_img = align_images(origin_img, completed_img)

            # 번호 위치 추출
            option_positions = extract_option_positions(page_num)

            if not option_positions:
                print(f"{page_num}페이지에서 번호를 찾을 수 없습니다.")
                continue

            # 사용자 마크 감지
            marked_options = detect_user_marks(aligned_img, option_positions, page_num)

            all_marked_options.extend(marked_options)
        except Exception as e:
            print(f"{page_num}페이지 처리 중 오류 발생: {str(e)}")
            continue

    if all_marked_options:
        # 결과를 데이터프레임으로 정리
        df = pd.DataFrame(all_marked_options)

        # 엑셀 파일로 저장
        df.to_excel("final_survey_results.xlsx", index=False)
        print("설문 결과가 final_survey_results.xlsx 파일로 저장되었습니다.")
    else:
        print("처리된 결과가 없습니다.")

if __name__ == "__main__":
    main()
