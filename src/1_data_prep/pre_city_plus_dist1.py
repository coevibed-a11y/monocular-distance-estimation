import json
import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool

# =========================================================
# [설정 영역] 경로 및 프로세스 수 설정
# =========================================================
# 3D 라벨 데이터 폴더 (LK_...json 파일들이 있는 곳)
DIR_3D_SOURCE = '/home/elicer/data/092_AD_City_Day/Validation/02_label_data/label_day_severe/3D' 

# [K 차량용] 파일명: calib_K_CF-L_calib.txt
CALIB_FILE_K = '/home/elicer/data/092_AD_City_Day/K_CF-L_calib.txt'

# [V 차량용] 파일명: calib_V_CF-L_calib.txt
CALIB_FILE_V = '/home/elicer/data/092_AD_City_Day/V_CF-L_calib.txt'

# 결과 저장 파일 경로 (Step 2에서 사용)
OUTPUT_PROJECTED_JSON = '/home/elicer/data/092_AD_City_Day/Validation/02_label_data/label_day_severe/projected_3d_bbox_only.json'
# CPU 코어 수 (시스템 상황에 맞게 조절, 보통 8개)
NUM_PROCESSES = 2 

# 이미지 해상도 (유효성 검사용)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
# =========================================================
# [전역 변수]
# =========================================================
GLOBAL_K_MAT_K = None
GLOBAL_RT_MAT_K = None
GLOBAL_K_MAT_V = None
GLOBAL_RT_MAT_V = None

def initializer(k_k, rt_k, k_v, rt_v):
    global GLOBAL_K_MAT_K, GLOBAL_RT_MAT_K, GLOBAL_K_MAT_V, GLOBAL_RT_MAT_V
    GLOBAL_K_MAT_K = np.array(k_k)
    GLOBAL_RT_MAT_K = np.array(rt_k)
    GLOBAL_K_MAT_V = np.array(k_v)
    GLOBAL_RT_MAT_V = np.array(rt_v)

# =========================================================
# [함수]
# =========================================================
def parse_calib(calib_path):
    if not os.path.exists(calib_path): return None, None
    with open(calib_path, 'r') as f: lines = f.readlines()
    k_mat, rt_mat = None, None
    for i, line in enumerate(lines):
        if "CameraExtrinsicMat" in line:
            vals = [float(x) for x in lines[i+1].strip().split(',')]
            rt_mat = np.array(vals).reshape(4, 4)
            try: rt_mat = np.linalg.inv(rt_mat)
            except: pass
        if "CameraMat" in line:
            vals = [float(x) for x in lines[i+1].strip().split(',')]
            k_mat = np.array(vals).reshape(3, 3)
    return k_mat, rt_mat

def project_3d_to_2d(points_3d, k_mat, rt_mat):
    if len(points_3d) == 0: return None
    n = len(points_3d)
    pts_homo = np.hstack((np.array(points_3d), np.ones((n, 1))))
    pts_cam = rt_mat @ pts_homo.T 
    pts_img_homo = k_mat @ pts_cam[:3, :]
    
    # Z값 1.0m 이상만 통과
    valid_indices = pts_img_homo[2, :] > 1.0
    if np.sum(valid_indices) < 3: return None
    
    pts_img_homo = pts_img_homo[:, valid_indices]
    u = pts_img_homo[0, :] / pts_img_homo[2, :]
    v = pts_img_homo[1, :] / pts_img_homo[2, :]
    return np.vstack((u, v)).T.astype(np.int32)

def is_valid_bbox(bbox):
    """[수정됨] 중심점 기반 엄격한 검사"""
    x, y, w, h = bbox
    if w <= 0 or h <= 0: return False

    # 중심점 계산
    cx = x + w / 2
    cy = y + h / 2
    
    # 중심이 화면 안에 있어야 함 (가장 확실한 방법)
    if cx < 0 or cx > IMG_WIDTH: return False
    if cy < 0 or cy > IMG_HEIGHT: return False

    # 화면의 80%를 넘는 거대 박스는 노이즈로 간주
    if w > IMG_WIDTH * 0.8 or h > IMG_HEIGHT * 0.8: return False
    
    return True

# =========================================================
# [단위 작업]
# =========================================================
def process_single_file(file_path_3d):
    filename = os.path.basename(file_path_3d)
    
    if "LK_" in filename: k_mat, rt_mat = GLOBAL_K_MAT_K, GLOBAL_RT_MAT_K
    elif "LV_" in filename: k_mat, rt_mat = GLOBAL_K_MAT_V, GLOBAL_RT_MAT_V
    else: k_mat, rt_mat = GLOBAL_K_MAT_K, GLOBAL_RT_MAT_K

    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    if len(parts) > 1 and (parts[0] == "LK" or parts[0] == "LV"):
        common_id = "_".join(parts[1:])
    else:
        common_id = name_no_ext

    try:
        with open(file_path_3d, 'r', encoding='utf-8') as f: data_3d = json.load(f)
    except: return None

    projected_objects = []
    if 'annotations' in data_3d:
        for obj3d in data_3d['annotations']:
            if '3D_points' in obj3d and len(obj3d['3D_points']) > 2:
                poly_2d_np = project_3d_to_2d(obj3d['3D_points'], k_mat, rt_mat)
                if poly_2d_np is not None:
                    x, y, w, h = cv2.boundingRect(poly_2d_np)
                    bbox = [int(x), int(y), int(w), int(h)]
                    
                    # [검사]
                    if is_valid_bbox(bbox):
                        raw_class = obj3d.get('class', '')
                        projected_objects.append({
                            'distance': float(obj3d.get('distance', -1.0)),
                            'class_name': raw_class.lower().strip(),
                            'bbox': bbox
                        })
    
    if not projected_objects: return None
    return {'id': common_id, 'objects': projected_objects}

# =========================================================
# [메인]
# =========================================================
def main_step1():
    print(f"🚀 Step 1: 듀얼 캘리브레이션 + 중심점 필터 (코어: {NUM_PROCESSES})...")
    k_k, rt_k = parse_calib(CALIB_FILE_K)
    k_v, rt_v = parse_calib(CALIB_FILE_V)
    
    if k_k is None or k_v is None: print("❌ Calib Error"); return

    files_3d = glob.glob(os.path.join(DIR_3D_SOURCE, '*.json'))
    print(f"📂 파일 수: {len(files_3d)}개")
    
    final_data = {}
    init_args = (k_k.tolist(), rt_k.tolist(), k_v.tolist(), rt_v.tolist())
    
    with Pool(processes=NUM_PROCESSES, initializer=initializer, initargs=init_args) as pool:
        results = pool.imap_unordered(process_single_file, files_3d)
        for res in tqdm(results, total=len(files_3d), desc="Projection"):
            if res is not None:
                final_data[res['id']] = res['objects']
                
    print(f"\n💾 저장 중...")
    with open(OUTPUT_PROJECTED_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False)
    print(f"✅ 완료: {OUTPUT_PROJECTED_JSON}")

if __name__ == "__main__":
    main_step1()