# config.py

# ==============================================================================
# [GROUP 1] Main 전용 설정
# (Main에서 영상 로드 및 루프 제어에만 사용되는 변수들)
# ==============================================================================
VIDEO_PATH = r"/home/elicer/dev/detectron2/Coach/city_drive_sample.mp4"

# 추론 및 시각화 해상도 (Main에서 cv2.resize 할 때 사용)
# TARGET_SIZE = (1280, 720) 
TARGET_SIZE = (1024, 576) 

# 시각화 프레임률 제한 (Main 루프 제어용)
VISUALIZATION_FPS = 20 


# ==============================================================================
# [GROUP 2] Visualizer 전용 설정 (Main + Visualizer 공유 포함)
# (화면에 어떻게 그릴지 스타일을 결정하는 변수들)
# ==============================================================================
# 색상 정의 (BGR)
COLOR_MAP = {
    "vehicle": (0, 255, 0),      # 초록
    "bus": (0, 200, 0),          # 진한 초록
    "truck": (0, 150, 0),        # 더 진한 초록
    "othercar": (50, 200, 50),   # 연두색
    
    "motorcycle": (0, 255, 255), # 노란색
    "bicycle": (0, 255, 255),    # 노란색
    
    "pedestrian": (0, 0, 255),   # 빨간색
    "rider": (0, 0, 200),        # 진한 빨강
    
    "freespace": (255, 100, 0)   # 파란색 계열
}

# 거리 정보를 텍스트로 표시할 클래스 목록
DIST_DISPLAY_CLASSES = [
    "vehicle", "bus", "truck", "othercar", 
    "motorcycle", "bicycle", "pedestrian", "rider"
]

# 최대 표시 거리 (이 거리 이상이면 숫자 표시 안 함)
MAX_DISPLAY_DIST = 50.0   


# ==============================================================================
# [GROUP 3] Detector 전용 설정 (Detector가 포함된 모든 공유 변수)
# (모델 로드, 추론, 거리 계산, 필터링, 추적 로직 등 핵심 알고리즘 변수들)
# ==============================================================================
# --- 3-1. 모델 경로 및 설정 ---
WEIGHT_PATH = r"/home/elicer/dev/01_script/3_eval_script/clear/model_best_dist.pth"
CONFIG_FILE = "/home/elicer/dev/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# --- 3-2. 클래스 정의 및 필터링 ---
THING_CLASSES = ["vehicle", "bus", "truck", "othercar", "motorcycle", "bicycle", "pedestrian", "rider", "trafficsign", "trafficlight", "constructionguide", "trafficdrum"]
STUFF_CLASSES = ["freespace", "curb", "sidewalk", "crosswalk", "roadmark", "whitelane", "yellowlane"]
ALL_CLASSES = THING_CLASSES + STUFF_CLASSES

# Detector가 감지 후 남길 클래스 목록 (여기 없는 건 Detector 단계에서 삭제됨)
TARGET_CLASSES = [
    "vehicle", "bus", "truck", "othercar", 
    "motorcycle", "bicycle", "pedestrian", "rider", 
    "freespace"
]

# --- 3-3. 거리 계산 튜닝 (Camera Geometry) ---
# TARGET_SIZE에 맞춰서 조정된 값들
HORIZON_Y = 256       # 소실점 Y좌표 / 720p 기준 320
GEO_SCALE = 800.0     # 위치 기반 거리 스케일 / 720p 기준 1000.0
FOCAL_LENGTH = 880.0  # 차폭 기반 거리 스케일 / 720p 기준 1100.0

# 차종별 실제 너비 (m)
REAL_WIDTH_MAP = {
    "vehicle": 1.85,    "othercar": 1.85,
    "bus": 2.20,        "truck": 2.20,
    "motorcycle": 0.80, "bicycle": 0.60,
    "pedestrian": 0.50, "rider": 0.60
}

# --- 3-4. 로직 임계값 (Logic Thresholds) ---
SIDE_MARGIN = 20          # 사이드 차량 판단 마진 (px)
SIDE_BOTTOM_RATIO = 0.80  # 진입 판단 높이 비율
ENTERING_THRESH = 2.5     # 진입 경고 거리 (m)
ROAD_MASK_CHECK = True    # 도로 위 객체 필터링 사용 여부

# --- 3-5. 추적(Tracking) 및 스무딩 ---
MATCH_DISTANCE_THRESH = 100.0  # 매칭 임계값 (px)
MAX_LOST_FRAMES = 5            # Ghost 유지 프레임
SMOOTH_ALPHA = 0.25            # 이동 평균 가중치