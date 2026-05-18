import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

# =========================================================
# [설정] 입력 데이터 경로 및 저장 설정
# =========================================================
INPUT_DATASETS = [
    {
        "name": "city",
        "path": "/home/elicer/data/092_AD_City_Day/Validation/02_label_data/lable_day_clear/val_city_coco_with_dist.json",
        "img_root": "/home/elicer/data/092_AD_City_Day/Validation/01_raw_data/image_day_clear" # 실제 이미지 경로
    },
    {
        "name": "highway",
        "path": "/home/elicer/data/090_AD_Hway_Day/Validation/02_label_data/train_coco_with_distance.json",
        "img_root": "/home/elicer/data/090_AD_Hway_Day/Validation/01_raw_data/image_data" # 실제 이미지 경로
    }
]

OUTPUT_DIR = '/home/elicer/data/090_AD_Hway_Day/Validation'
MIN_OBJECTS = 30  # 검증 셋에 보장할 클래스별 최소 객체 수
VAL_RATIO = 0.5   # 전체 Val 데이터를 (New Val : Test)로 나누는 비율 (0.5 = 50:50)

# =========================================================
# [함수]
# =========================================================
def create_base_coco():
    return {
        "info": {"description": "Integrated Val/Test Dataset", "year": 2023},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("🚀 데이터셋 통합 및 분할 시작...")

    # 1. 통합 카테고리 맵 생성 (이름 기준)
    # 모든 데이터셋의 카테고리를 읽어서 공통 ID를 부여합니다.
    unified_categories = {} # {"car": 1, "bus": 2 ...}
    next_cat_id = 1
    
    # 데이터 임시 저장소
    merged_images = []
    merged_annotations = []
    
    # ID 충돌 방지용 오프셋
    img_id_offset = 0
    ann_id_offset = 0

    print("\n🔹 [Step 1] 데이터 병합 및 ID 재발급")
    
    for dset in INPUT_DATASETS:
        name = dset['name']
        path = dset['path']
        img_root = dset['img_root']
        
        if not os.path.exists(path):
            print(f"⚠️ Warning: 파일이 없습니다. 건너뜁니다: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # A. 카테고리 매핑 테이블 생성 (Local ID -> Unified ID)
        local_to_unified = {}
        for cat in data.get('categories', []):
            cat_name = cat['name'].lower().strip().replace(" ", "")
            
            # 통합 맵에 없으면 새로 등록
            if cat_name not in unified_categories:
                unified_categories[cat_name] = {
                    "id": next_cat_id,
                    "name": cat['name'], # 원본 이름 유지
                    "supercategory": cat.get('supercategory', 'none')
                }
                next_cat_id += 1
            
            local_to_unified[cat['id']] = unified_categories[cat_name]['id']

        # B. 이미지 및 어노테이션 통합
        print(f"   - {name}: 이미지 {len(data['images'])}장 처리 중...")
        
        current_max_img_id = 0
        current_max_ann_id = 0
        
        for img in data['images']:
            new_img = img.copy()
            # ID 재발급
            new_img['id'] += img_id_offset
            # 파일 경로 절대경로로 변환 (학습 때 편하게)
            # file_name이 이미 절대경로면 두고, 아니면 root와 결합
            if not new_img['file_name'].startswith('/'):
                new_img['file_name'] = os.path.join(img_root, new_img['file_name'])
            
            # 데이터셋 출처 표시 (나중에 구분하고 싶을 때 유용)
            new_img['dataset_source'] = name
            
            merged_images.append(new_img)
            current_max_img_id = max(current_max_img_id, img['id'])

        for ann in data['annotations']:
            new_ann = ann.copy()
            # ID 재발급
            new_ann['id'] += ann_id_offset
            new_ann['image_id'] += img_id_offset
            # 카테고리 ID 변환
            new_ann['category_id'] = local_to_unified[ann['category_id']]
            
            merged_annotations.append(new_ann)
            current_max_ann_id = max(current_max_ann_id, ann['id'])
            
        # 다음 데이터셋을 위해 오프셋 업데이트 (충분히 띄움)
        img_id_offset += (current_max_img_id + 10000)
        ann_id_offset += (current_max_ann_id + 10000)

    # 통합된 카테고리 리스트 생성
    final_categories = [v for k, v in unified_categories.items()]
    final_categories.sort(key=lambda x: x['id'])
    
    print(f"   -> 병합 완료: 총 이미지 {len(merged_images)}장, 라벨 {len(merged_annotations)}개")
    print(f"   -> 통합 카테고리 수: {len(final_categories)}개")

    # ---------------------------------------------------------
    # [Step 2] Stratified Split (균형 분할)
    # ---------------------------------------------------------
    print("\n🔹 [Step 2] 균형 분할 (New Val vs Test)")
    
    img_id_to_anns = defaultdict(list)
    img_id_to_cats = defaultdict(set)
    
    for ann in merged_annotations:
        img_id_to_anns[ann['image_id']].append(ann)
        img_id_to_cats[ann['image_id']].add(ann['category_id'])
        
    # 희귀 클래스 파악
    cat_counts = defaultdict(int)
    for ann in merged_annotations:
        cat_counts[ann['category_id']] += 1
        
    sorted_cats = sorted(final_categories, key=lambda c: cat_counts[c['id']])
    
    # 분할 컨테이너
    split_sets = {"val": set(), "test": set()}
    split_counts = {"val": defaultdict(int), "test": defaultdict(int)}
    used_img_ids = set()
    
    # 해당 클래스를 가진 이미지 맵
    cat_to_imgs = defaultdict(list)
    for img in merged_images:
        iid = img['id']
        if iid in img_id_to_cats:
            for cid in img_id_to_cats[iid]:
                cat_to_imgs[cid].append(iid)

    # 1. 최소 수량 보장 할당
    print(f"   -> 최소 수량({MIN_OBJECTS}개) 확보 중...")
    for cat in sorted_cats:
        cid = cat['id']
        candidates = [i for i in cat_to_imgs[cid] if i not in used_img_ids]
        random.seed(42)
        random.shuffle(candidates)
        
        for iid in candidates:
            # Val, Test 중 부족한 곳으로 보냄
            val_full = split_counts['val'][cid] >= MIN_OBJECTS
            test_full = split_counts['test'][cid] >= MIN_OBJECTS
            
            target = None
            if not val_full: target = "val"
            elif not test_full: target = "test"
            else: continue # 둘 다 찼으면 패스
            
            split_sets[target].add(iid)
            used_img_ids.add(iid)
            
            # 카운트 갱신
            for c in img_id_to_cats[iid]:
                cnt = len([a for a in img_id_to_anns[iid] if a['category_id'] == c])
                split_counts[target][c] += cnt

    # 2. 나머지 랜덤 할당
    print("   -> 나머지 데이터 배정...")
    remaining = [img['id'] for img in merged_images if img['id'] not in used_img_ids]
    random.shuffle(remaining)
    
    # VAL_RATIO에 맞춰 분배
    target_val_count = int(len(merged_images) * VAL_RATIO)
    
    for iid in remaining:
        if len(split_sets['val']) < target_val_count:
            target = "val"
        else:
            target = "test"
        
        split_sets[target].add(iid)

    # ---------------------------------------------------------
    # [Step 3] 파일 저장
    # ---------------------------------------------------------
    print("\n🔹 [Step 3] 최종 JSON 저장")
    img_lookup = {img['id']: img for img in merged_images}
    
    for split_name in ["val", "test"]:
        new_data = create_base_coco()
        new_data['categories'] = final_categories
        
        target_ids = split_sets[split_name]
        
        for iid in target_ids:
            new_data['images'].append(img_lookup[iid])
            if iid in img_id_to_anns:
                new_data['annotations'].extend(img_id_to_anns[iid])
        
        filename = f"integrated_{split_name}.json"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=None, ensure_ascii=False)
            
        print(f"   ✅ [{split_name.upper()}] 저장 완료: {save_path} (이미지 {len(new_data['images'])}장)")

    print("\n🎉 모든 작업 완료!")

if __name__ == "__main__":
    main()