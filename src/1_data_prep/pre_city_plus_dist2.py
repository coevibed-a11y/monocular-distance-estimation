import json
import os
import math
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict

# =========================================================
# [설정]
# =========================================================
COCO_JSON_PATH = '/home/elicer/data/092_AD_City_Day/Training/02_label_data/lable_day_clear/city_data_coco.json'  # 원본 COCO 형식 파일
PROJECTED_JSON_PATH = '/home/elicer/data/092_AD_City_Day/Training/02_label_data/lable_day_clear/projected_3d_bbox_only.json' # Step 1 결과물
OUTPUT_JSON_PATH = '/home/elicer/data/092_AD_City_Day/Training/02_label_data/lable_day_clear/train_city_coco_with_dist.json'

IOU_THRESHOLD = 0.5 
NUM_PROCESSES = 2

# =========================================================
# [함수] 산술 IoU
# =========================================================
def calculate_bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea
    
    if unionArea == 0: return 0.0
    return interArea / unionArea

# =========================================================
# [단위 작업] 
# =========================================================
def merge_chunk(args):
    img_ids, coco_anns_map, proj_data, id_to_name, mapping_rule = args
    updated_anns = []
    matched_cnt = 0

    for img_id in img_ids:
        target_anns = coco_anns_map.get(img_id, [])
        if not target_anns: continue
        
        filename = id_to_name.get(img_id)
        if not filename: 
            updated_anns.extend(target_anns); continue

        # ID 추출 (파일명 기반)
        pure_name = os.path.basename(filename)
        name_no_ext = os.path.splitext(pure_name)[0]
        common_id = name_no_ext.replace("CK_", "")
        if common_id.endswith("_F"): common_id = common_id[:-2]
        if common_id.endswith(".png"): common_id = common_id[:-4]

        # 3D 데이터 가져오기
        proj_objs = proj_data.get(common_id, [])
        
        # 3D 데이터가 없으면 초기화 후 패스
        if not proj_objs:
            for t_ann in target_anns:
                t_ann['distance'] = -1.0
            updated_anns.extend(target_anns)
            continue
        
        for t_ann in target_anns:
            t_cat_id = t_ann['category_id']
            box_2d = t_ann.get('bbox')
            
            if not box_2d: 
                t_ann['distance'] = -1.0
                continue
            
            best_iou = 0
            best_dist = -1.0
            
            for p_obj in proj_objs:
                # [핵심] 동적 매핑 적용
                # 3D Class Name (예: "Traffic Sign") -> 정제 -> "trafficsign"
                p_class_raw = p_obj.get('class_name', '')
                # 공백 제거 및 소문자화 (매핑 확률 높이기)
                p_key = p_class_raw.replace(" ", "").lower()
                
                # 매핑 테이블에서 ID 조회
                mapped_cat_id = mapping_rule.get(p_key)
                
                # 매핑 실패 시 "car"나 "vehicle" 같은 동의어 한번 더 체크 (안전장치)
                if mapped_cat_id is None:
                    if "car" in p_key: mapped_cat_id = mapping_rule.get("vehicle") or mapping_rule.get("car")
                    elif "truck" in p_key: mapped_cat_id = mapping_rule.get("truck")
                    elif "bus" in p_key: mapped_cat_id = mapping_rule.get("bus")
                
                # 클래스가 다르면 Skip
                if mapped_cat_id is None or mapped_cat_id != t_cat_id:
                    continue

                # 클래스가 같을 때만 IoU 계산
                iou = calculate_bbox_iou(box_2d, p_obj['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_dist = p_obj['distance']
            
            if best_iou >= IOU_THRESHOLD:
                t_ann['distance'] = float(best_dist)
                matched_cnt += 1
            else:
                t_ann['distance'] = -1.0
        
        updated_anns.extend(target_anns)
        
    return updated_anns, matched_cnt

# =========================================================
# [메인]
# =========================================================
def main_step2():
    print(f"🚀 Step 2: 자동 ID 매핑 & 병합 (코어: {NUM_PROCESSES})...")
    
    with open(COCO_JSON_PATH, 'r') as f: coco_data = json.load(f)
    with open(PROJECTED_JSON_PATH, 'r') as f: proj_data = json.load(f)
    
    # [핵심] 카테고리 자동 매핑 테이블 생성
    # 2D 데이터의 categories를 읽어서 { "vehicle": 1, "bus": 2 ... } 맵을 만듭니다.
    print(">>> 카테고리 ID 자동 분석 중...")
    mapping_rule = {}
    for cat in coco_data.get('categories', []):
        # 이름 정규화 (소문자, 공백제거) : "Traffic Sign" -> "trafficsign"
        clean_name = cat['name'].replace(" ", "").lower()
        mapping_rule[clean_name] = cat['id']
        # 원본 이름도 키로 추가 (안전장치)
        mapping_rule[cat['name'].lower()] = cat['id']
        
    print(f"📋 감지된 매핑 규칙: {mapping_rule}")

    all_img_ids = [img['id'] for img in coco_data['images']]
    id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
        
    # 병렬 처리 분배
    chunk_size = math.ceil(len(all_img_ids) / NUM_PROCESSES)
    id_chunks = [all_img_ids[i:i+chunk_size] for i in range(0, len(all_img_ids), chunk_size)]
    
    # 매핑 룰도 함께 전달
    tasks = [(chunk, img_to_anns, proj_data, id_to_name, mapping_rule) for chunk in id_chunks]
    
    final_anns = []
    total_matched = 0
    
    with Pool(NUM_PROCESSES) as pool:
        results = pool.imap(merge_chunk, tasks)
        for res_anns, cnt in tqdm(results, total=len(tasks), desc="Matching"):
            final_anns.extend(res_anns)
            total_matched += cnt
            
    # 최종 저장 (순서는 바뀌지만 ID로 연결되므로 안전)
    coco_data['annotations'] = final_anns
    print(f"\n✅ 완료! 매칭 성공: {total_matched}건")
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)
    print(f"💾 저장: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main_step2()