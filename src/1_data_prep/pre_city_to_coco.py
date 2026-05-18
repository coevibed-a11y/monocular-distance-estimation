import json
import os
import glob
from tqdm import tqdm

# ==========================================
# 1. 설정 (경로 확인 필수)
# ==========================================
# 원본 라벨 파일들이 들어있는 폴더 경로
INPUT_DIR = '/home/elicer/data/092_AD_City_Day/Training/02_label_data/label_day_severe/2D'

# 결과가 저장될 파일 이름
OUTPUT_FILE = '/home/elicer/data/092_AD_City_Day/Training/02_label_data/label_day_severe/city_data_coco.json'

# ==========================================
# 2. 타겟 카테고리 (19개)
# ==========================================
TARGET_CATEGORIES = [
    {"id": 1, "name": "vehicle", "is_thing": 1},
    {"id": 2, "name": "bus", "is_thing": 1},
    {"id": 3, "name": "truck", "is_thing": 1},
    {"id": 4, "name": "othercar", "is_thing": 1},
    {"id": 5, "name": "motorcycle", "is_thing": 1},
    {"id": 6, "name": "bicycle", "is_thing": 1},
    {"id": 7, "name": "pedestrian", "is_thing": 1},
    {"id": 8, "name": "rider", "is_thing": 1},
    {"id": 9, "name": "trafficsign", "is_thing": 1},
    {"id": 10, "name": "trafficlight", "is_thing": 1},
    {"id": 11, "name": "constructionguide", "is_thing": 1},
    {"id": 12, "name": "trafficdrum", "is_thing": 1},
    {"id": 13, "name": "freespace", "is_thing": 0},
    {"id": 14, "name": "curb", "is_thing": 0},
    {"id": 15, "name": "sidewalk", "is_thing": 0},
    {"id": 16, "name": "crosswalk", "is_thing": 0},
    {"id": 17, "name": "roadmark", "is_thing": 0},
    {"id": 18, "name": "whitelane", "is_thing": 0},
    {"id": 19, "name": "yellowlane", "is_thing": 0}
]

# ==========================================
# 3. 매핑 규칙
# ==========================================
MAPPING_RULE = {
    "car": 1, "car-b": 1,
    "bus": 2, "bus-b": 2,
    "truck": 3, "truck-b": 3, "truckbus": 3, "truckbus-b": 3,
    "othercar": 4, "policecar": 4, "ambulance": 4, "schoolbus": 4,
    "motorcycle": 5, "two-wheel vehicle": 5, "two-wheel vehicle-b": 5, "twowheeler": 5,
    "bicycle": 6,
    "pedestrian": 7, "pedestrian-b": 7, "adult": 7, "kid student": 7,
    "rider": 8,
    "trafficsign": 9, "traffic sign": 9,
    "trafficlight": 10, "traffic light": 10,
    "constructionguide": 11,
    "trafficdrum": 12,
    "freespace": 13,
    "curb": 14,
    "sidewalk": 15,
    "crosswalk": 16,
    "roadmark": 17, "speed bump": 17, "parking space": 17,
    "safetyzone": 17, "stoplane": 17, "bluelane": 17, "redlane": 17,
    "whitelane": 18,
    "yellowlane": 19
}

def main():
    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    print(f"🔄 총 {len(json_files)}개의 라벨 파일을 찾았습니다.")

    # images를 맨 앞으로 배치
    final_data = {
        "images": [],
        "annotations": [],
        "categories": TARGET_CATEGORIES,
        "info": {"description": "Integrated Dataset (Custom 19 Classes)", "year": 2023},
        "licenses": []
    }

    new_image_id = 0
    new_annotation_id = 0
    mapped_count = 0
    skipped_count = 0

    print("🚀 데이터 변환 및 포맷 수정 시작...")

    for file_path in tqdm(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 로컬 카테고리 매핑 생성
            local_id_to_name = {}
            source_cats = data.get('category', data.get('categories', []))
            for cat in source_cats:
                local_id_to_name[cat['id']] = cat['name']

            # 이미지 처리
            for img in data['images']:
                clean_file_name = os.path.basename(img['file_name'])
                
                final_data['images'].append({
                    "id": new_image_id,
                    "file_name": clean_file_name,
                    "height": img['height'],
                    "width": img['width']
                })
                
                current_file_img_id = img['id']

                # 어노테이션 처리
                if 'annotations' in data:
                    for ann in data['annotations']:
                        if ann['image_id'] != current_file_img_id:
                            continue
                        
                        local_cat_id = ann['category_id']
                        if local_cat_id not in local_id_to_name:
                            continue
                            
                        cat_name_raw = local_id_to_name[local_cat_id]
                        cat_name_lower = cat_name_raw.lower().strip()
                        
                        if cat_name_lower in MAPPING_RULE:
                            target_cat_id = MAPPING_RULE[cat_name_lower]
                            
                            # --- [수정 1] Segmentation 좌표 변환 로직 ---
                            new_segmentation = []
                            raw_seg = ann.get('segmentation', [])
                            
                            # Case A: 원본이 {"coord": {"points": ...}} 형태일 때 (Superb AI 포맷)
                            if isinstance(raw_seg, dict) and 'coord' in raw_seg:
                                points_outer = raw_seg['coord']['points']
                                # points_outer는 보통 [[[{"x":.., "y":..}]]] 형태의 3중 리스트 구조임
                                for shape in points_outer: 
                                    for inner_shape in shape: # 한 번 더 들어가서
                                        poly_coords = []
                                        for pt in inner_shape: # 실제 점들의 리스트 순회
                                            if isinstance(pt, dict) and 'x' in pt and 'y' in pt:
                                                # x, y 순서대로 float로 변환하여 추가
                                                poly_coords.append(float(pt['x']))
                                                poly_coords.append(float(pt['y']))
                                        
                                        # 유효한 좌표가 있으면 추가
                                        if poly_coords:
                                            new_segmentation.append(poly_coords)
                                            
                            # Case B: 이미 리스트 형태라면 그대로 복사하되 float 변환 체크
                            elif isinstance(raw_seg, list):
                                new_segmentation = raw_seg 

                            # --- [수정 2] Area 및 BBox를 Float(.0)으로 변환 ---
                            area_float = float(ann.get('area', 0))
                            bbox_float = [float(x) for x in ann.get('bbox', [])]

                            final_data['annotations'].append({
                                "id": new_annotation_id,
                                "image_id": new_image_id,
                                "category_id": target_cat_id,
                                "bbox": bbox_float,          # Float 리스트
                                "area": area_float,          # Float 값 (예: 894720.0)
                                "segmentation": new_segmentation, # [[x,y,x,y...]] 형태
                                "iscrowd": ann.get('iscrowd', 0),
                                "distance": -1
                            })
                            new_annotation_id += 1
                            mapped_count += 1
                        else:
                            skipped_count += 1
                
                new_image_id += 1

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    print("💾 파일 저장 중... (indent 적용)")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print("\n✅ 변환 완료!")
    print(f"📁 저장 파일: {OUTPUT_FILE}")
    print(f"📊 총 이미지 수: {len(final_data['images'])}")
    print(f"🏷️ 총 어노테이션 수: {len(final_data['annotations'])}")

if __name__ == "__main__":
    main()