import os
from detectron2.config import get_cfg
from detectron2 import model_zoo

# =========================================================
# [설정] 모델이 저장된 폴더 경로를 여기에 적어주세요!
# =========================================================
OUTPUT_DIR = "/home/elicer/dev/gt/data/model/20251126_145642" # <--- 여기 수정!

def create_config():
    print(f"🚀 Config 파일 생성 시작: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Error: 해당 폴더가 존재하지 않습니다: {OUTPUT_DIR}")
        return

    # 1. 기본 Config 로드 (학습 때 쓴 것과 동일한 베이스)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # 2. 우리가 변경한 하이퍼파라미터 덮어쓰기
    # (train_model_final_v3.py의 설정과 똑같이 맞춰줍니다)
    
    # 데이터셋 이름 (나중에 추론할 땐 중요하지 않지만 형식상 넣어둠)
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ("my_val",)
    
    # 모델 구조 설정
    cfg.MODEL.ROI_HEADS.NAME = "DistanceROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19 # 클래스 개수 (필수!)
    
    # 학습 파라미터 (추론 시엔 큰 영향 없지만 기록용)
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.STEPS = (28000, 36000)
    
    # 출력 경로
    cfg.OUTPUT_DIR = OUTPUT_DIR
    
    # 3. 파일 저장
    save_path = os.path.join(OUTPUT_DIR, "config.yaml")
    with open(save_path, "w") as f:
        f.write(cfg.dump())
        
    print(f"✅ Config 파일 저장 완료: {save_path}")
    print("   -> 이제 detector.py에서 이 파일을 로드할 수 있습니다.")

if __name__ == "__main__":
    create_config()