import cv2
import config as cfg
from detector import VehicleDetector

IMG_PATH = r"/home/elicer/dev/gt/data/test_data/371_ND_000_FC_176.jpg"

def main():
    detector = VehicleDetector()
    
    # 이미지 한 장 로드 (테스트용)
    img_path = IMG_PATH
    cap = cv2.VideoCapture(img_path)
    ret, img = cap.read()
    cap.release() # VideoCapture 해제
    if not ret:
        print(f"Failed to read image: {img_path}")
        return
    
    # detector 내부에서 resize를 수행하므로 여기서는 주석 처리하거나 제거해도 됩니다.
    # 만약 원본 이미지에 그리기를 원한다면 이 줄을 제거하세요.
    # img = cv2.resize(img, cfg.TARGET_SIZE) 

    # ----------------------------------------------------------
    # [사용 예시] 시각화 담당자가 쓰게 될 코드 패턴
    # ----------------------------------------------------------
    results, road_mask = detector.run(img)
    
    # 데이터 확인용 출력 (필요 시 주석 해제)
    # print("Classes:", results['class'])
    # print("Distances:", results['distance'])
    # print("Entering:", results['is_entering'])
    # print("Boxes:", results['box'])
    # print("Scores:", results['score'])
    
    # ==========================================================
    # [시각화 코드 추가]
    # ==========================================================
    CONF_THRESHOLD = 0.5 # 신뢰도 임계값 설정
    vis_img = img.copy() # 원본 이미지 보존을 위해 복사본 사용

    # zip으로 묶어서 한 번에 처리
    for c_name, dist, is_ent, box, score in zip(results['class'], results['distance'], results['is_entering'], results['box'], results['score']):
        
        # 1. 신뢰도 필터링
        if score < CONF_THRESHOLD:
            continue
            
        # 2. freespace 제외 (필요에 따라 조정)
        if c_name == "freespace":
            continue

        # 3. 좌표 정수 변환
        x1, y1, x2, y2 = map(int, box)
        
        # 4. 색상 설정 (예: 진입 차량은 빨간색, 나머지는 초록색)
        color = (0, 0, 255) if is_ent else (0, 255, 0) # BGR 순서

        # 5. 바운딩 박스 그리기
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # 6. 텍스트 정보 생성 (클래스명 + 거리)
        # 거리가 유효한 경우에만 표시 (예: 0 이상)
        dist_text = f"{dist:.1f}m" if dist > 0 else ""
        label = f"{c_name} {dist_text} ({score:.2f})"
        
        # 7. 텍스트 그리기 (배경 박스 추가로 가독성 높임)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + w, y1), color, -1) # 텍스트 배경
        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1) # 흰색 글씨
        
        # 진입 차량 표시 (선택 사항)
        if is_ent:
            cv2.putText(vis_img, "ENTERING", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2) # 박스 하단에 표시

    # 8. 결과 이미지 저장 및 확인
    save_path = "result_vis.jpg"
    cv2.imwrite(save_path, vis_img)
    print(f"Visualization saved to: {save_path}")
    
    # (옵션) 서버 환경이 아닐 경우 창으로 띄우기
    # cv2.imshow("Result", vis_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()