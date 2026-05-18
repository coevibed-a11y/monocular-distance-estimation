# Monocular Distance Estimation (단일 카메라 기반 차량 거리 추정 및 위험 감지 시스템)

이 프로젝트는 자율주행 및 ADAS(첨단 운전자 보조 시스템) 환경에서 단일 카메라(Monocular Camera)만으로 전방 객체의 거리를 추정하고 위험(Cut-in 등)을 판단하는 비전 시스템입니다. 

기존의 단순 Bounding Box 검출을 넘어, **AI 모델의 회귀(Regression) 예측과 기하학적(Geometry) 연산을 결합한 하이브리드 로직**을 통해 안정적인 거리 데이터를 추출합니다.

### 📊 System Overview
* **Input:** 단일 RGB 카메라 영상 (Monocular Camera Stream)
* **Output:** - 실시간 2D 객체 검출 (Object Detection)
  - 단일 카메라 기반 종방향 상대 거리 추정 (Distance Regression)
  - 인접 차선 차량의 컷인(Cut-in) 위험도 판별
  - 2D 이미지 좌표의 BEV(Bird's Eye View) 공간 투영 및 실시간 시각화

## 📌 주요 기능 (Key Features)
* **Custom ROI Head (DistanceROIHeads):** Detectron2 기반 Mask R-CNN 구조를 확장하여, Bounding Box 예측과 동시에 거리를 직접 추론(Regression)하도록 Head 커스터마이징.
* **Hybrid Distance Logic:** 모델이 예측한 거리(AI)와 카메라 캘리브레이션/소실점 기반 거리(Math)를 융합하여 오차 최소화.
* **Cut-in (끼어들기) 감지:** 주행 코리더(Safety Corridor)와 객체 중심점, Mask IOU를 분석하여 측면 차량의 차선 진입(Entering) 여부 판별.
* **실시간 대시보드 (Live Demo):** FastAPI 기반 웹 스트리밍, BEV(Bird's Eye View) 미니맵, 3단계 위험 경고 (Danger / Caution / Safe) 시각화.

### 🛠️ Key Engineering Contributions (Main Developer)
* **이기종 데이터 정합 파이프라인 독자 구축**
  - 3D Point Cloud/라벨 데이터를 카메라 캘리브레이션 매트릭스($Extrinsic \times Intrinsic$) 기하 연산을 통해 2D 평면으로 정밀 투영(Projection)하는 파이프라인 구현.
  - 센서 간 오차 보정을 위해 IoU 0.5 임계값 기반의 데이터 정합(Data Association) 알고리즘을 설계하여 2D 데이터 내 실제 거리($Distance$) 값 무결성 주입.
* **Detectron2 Custom ROI Head 개전 및 커스텀 로스 설계**
  - 국소 영역 특징 추정 강화를 위해 Detectron2 구조 내부의 ROI Head를 수정하여 거리 추정 전용 Regression 헤드를 추가 바인딩하고 Loss 함수 커스텀 설계.
* **실시간 스트리밍 라이브 데모 및 시스템화**
  - 학습 완료된 딥러닝 체크포인트를 FastAPI 백엔드 아키텍처와 연동하여 저지연 실시간 스트리밍(Live Demo) 환경 및 웹 기반 시각화 파이프라인 검증 완료.

### 📈 Performance Metrics
* **Inference Speed:** 평균 `XX FPS` (구동 환경: 로컬 GPU RTX XXXX 기준)
* **Distance Estimation Accuracy (MAE):** - 단거리(0~10m) 구간: 평균 `X.X m` 오차 범위 이내
  - 중거리(10~30m) 구간: 평균 `X.X m` 오차 범위 이내

## 📂 디렉토리 구조 (Directory Structure)
```text
monocular-distance-estimation/
├─ configs/           # 모델 아키텍처 및 하이퍼파라미터 설정 (config.yaml)
├─ weights/           # 학습된 모델 가중치 저장소 (.pth)
├─ results/           # 모델 검증 결과 및 평가지표 (MAE 등)
└─ src/               # 전체 소스 코드
   ├─ 1_data_prep/    # 3D 라벨 -> 2D Projection 및 COCO 포맷 변환 로직
   ├─ 2_training/     # Detectron2 기반 Custom Head 학습 파이프라인
   ├─ 3_evaluation/   # 구간별 거리 오차(MAE) 측정 및 테스트
   └─ 4_live_demo/    # FastAPI 및 시각화(Visualizer) 서빙 모듈
```

## 📊 사용 데이터 (Data)
* **도심 및 고속도로 주행 데이터** (맑은 날 / 악천후 조건)
* 원본 3D Bounding Box 라벨링 데이터에서 Camera Intrinsic/Extrinsic 매트릭스를 활용해 2D Box로 투영(Projection) 후, Z축 거리값을 `distance` 필드로 매핑하여 COCO JSON 형태로 구축.
* 19개 분류를 9개의 Target Class(vehicle, bus, truck, pedestrian 등) 및 freespace(주행 가능 영역)로 정제하여 사용.

## 🚀 실행 방법 (How to Run)

**1. 환경 설정**
```bash
pip install -r requirements.txt
# Detectron2는 공식 문서를 참고하여 환경(PyTorch/CUDA)에 맞게 설치
```

**2. 모델 학습 (Training)**
```bash
python src/2_training/train_model_clear.py
```

**3. 실시간 데모 실행 (Live Streaming)**
```bash
python src/4_live_demo/jh_fapi.py
# 실행 후 브라우저에서 http://localhost:8000 접속 (스페이스바로 재생/일시정지 제어)
```

## 📸 결과 및 시각화 (Results & Visualization)
*(여기에 라이브 데모 화면 캡처 이미지나, BEV 미니맵이 작동하는 GIF를 1~2장 정도 첨부하면 포트폴리오로서 완성도가 크게 올라갑니다.)*
* **[이미지 1 첨부 자리: 객체 인식 및 거리 텍스트 표시 화면]**
* **[이미지 2 첨부 자리: 하단 상태바 및 우측 하단 BEV 맵 표시 화면]**

## 🛠 향후 개선 과제 (Future Work)
* **경로 동적 할당 (Dynamic Path Handling):** 현재 하드코딩된 로컬 절대 경로(`/home/...`)를 `pathlib` 모듈을 활용하여 프로젝트 루트 기준의 상대 경로로 실행되도록 리팩토링.
* **공통 모듈 통합 (Code Modularization):** `clear`, `severe`, `live_demo` 파트에 중복되어 있는 코어 클래스(`VehicleDetector`)와 설정 파일(`config.py`)을 `src/core/` 디렉토리로 통합하여 유지보수성 향상.