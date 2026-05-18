import json
import os

# 확인하고 싶은 파일 경로 (현재 학습에 쓰고 있는 검증 파일)
TARGET_JSON = '/home/elicer/data/090_AD_Hway_Day/Validation/integrated_val.json'

def main():
    print(f"🔍 데이터 검사 시작: {TARGET_JSON}")
    
    if not os.path.exists(TARGET_JSON):
        print("❌ 파일이 없습니다.")
        return

    with open(TARGET_JSON, 'r') as f:
        data = json.load(f)
    
    total_anns = len(data['annotations'])
    valid_dist_count = 0
    
    print(f"📊 총 어노테이션 수: {total_anns}")
    
    for ann in data['annotations']:
        dist = ann.get('distance', -1)
        # 거리가 0보다 크면 유효한 데이터
        if dist > 0:
            valid_dist_count += 1
            
    print(f"✅ 유효한 Distance(>0) 개수: {valid_dist_count}")
    
    if valid_dist_count == 0:
        print("\n🚨 [진단 결과] 이 파일에는 거리 정보가 전혀 없습니다!")
        print("   -> create_integrated_split.py 실행 시 입력 파일 경로를 잘못 지정했을 확률이 높습니다.")
        print("   -> 아까 매칭에 성공했던 'city_data_coco_final_v3.json'을 사용했는지 확인하세요.")
    else:
        print(f"\n✨ [진단 결과] {valid_dist_count}개의 거리 데이터가 있습니다. (비율: {valid_dist_count/total_anns*100:.1f}%)")

if __name__ == "__main__":
    main()