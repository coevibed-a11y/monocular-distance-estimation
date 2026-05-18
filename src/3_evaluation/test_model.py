import os
import cv2
import torch
import json
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy


# Detectron2 Imports
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import pairwise_iou, BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import pairwise_iou, BoxMode, Boxes
from torch import nn

# =========================================================
# [설정]
# =========================================================
TEST_JSON_PATH = '/home/elicer/data/091_AD_Severe_Day/Validation/integrated_test.json'
MODEL_PATH = '/home/elicer/dev/gt/custom_model/model_best_dist.pth'
OUTPUT_DIR = '/home/elicer/dev/01_script/3_eval_script/severe/eval_results_detail'
OS_ENV_SETUP = True # 멈춤 방지 설정 적용 여부

# =========================================================
# [0. 환경 설정]
# =========================================================
if OS_ENV_SETUP:
    os.environ["TORCH_Dynamo"] = "disable"
    os.environ["TORCH_INDUCTOR"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cv2.setNumThreads(0)

# =========================================================
# [1. 모델 정의]
# =========================================================
@ROI_HEADS_REGISTRY.register()
class DistanceROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        input_dim = self.box_head.output_shape.channels if hasattr(self.box_head, 'output_shape') else 1024
        self.distance_fc = nn.Sequential(nn.Linear(input_dim, 1), nn.ReLU())
        self.max_distance = 100.0

    def _forward_box(self, features, proposals):
        features_list = [features[f] for f in self.box_in_features]
        pred_instances, _ = self.box_predictor.inference(
            self.box_predictor(self.box_head(self.box_pooler(features_list, [x.proposal_boxes for x in proposals]))), 
            proposals
        )
        if len(pred_instances) == 0: return pred_instances
        
        pred_boxes = [x.pred_boxes for x in pred_instances]
        final_box_features = self.box_pooler(features_list, pred_boxes)
        final_box_features = self.box_head(final_box_features)
        pred_normalized = self.distance_fc(final_box_features)
        final_distances = pred_normalized * self.max_distance 
        
        start_idx = 0
        for instances in pred_instances:
            num_boxes = len(instances)
            instances.pred_distances = final_distances[start_idx : start_idx + num_boxes]
            start_idx += num_boxes
        return pred_instances

# =========================================================
# [2. 데이터 로드 & 매퍼]
# =========================================================
def get_val_dicts(json_path):
    print(f"📂 Loading Data: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    categories = sorted(data['categories'], key=lambda x: x['id'])
    thing_classes = [c['name'] for c in categories]
    
    img_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    dataset_dicts = []
    for img in tqdm(data['images'], desc="Parsing"):
        record = {}
        if os.path.isabs(img['file_name']): record["file_name"] = img['file_name']
        else: continue 
        if not os.path.exists(record["file_name"]): continue

        record["image_id"] = img['id']
        record["height"] = img['height']
        record["width"] = img['width']
        
        objs = []
        for ann in img_to_anns.get(img['id'], []):
            dist_val = ann.get("distance", -1.0)
            if dist_val is None: dist_val = -1.0
            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS, 
                "category_id": ann["category_id"] - 1, 
                "distance": float(dist_val),
                "iscrowd": ann.get("iscrowd", 0)
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts, thing_classes

def test_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    try: image = utils.read_image(dataset_dict["file_name"], format="BGR")
    except: return None
    transform_list = [T.Resize((800, 800))]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    if "annotations" in dataset_dict:
        annos = [utils.transform_instance_annotations(obj, transforms, image.shape[:2]) for obj in dataset_dict.pop("annotations")]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        MAX_DIST = 100.0
        normalized_dists = []
        for obj in annos:
            d = obj.get("distance", -1.0)
            if d != -1.0: d = min(d, MAX_DIST) / MAX_DIST
            normalized_dists.append(d)
        instances.gt_distances = torch.tensor(normalized_dists, dtype=torch.float32)
        dataset_dict["instances"] = instances
    return dataset_dict

# =========================================================
# [3. 메인 분석기]
# =========================================================
class DeepEvaluator:
    def __init__(self, cfg, val_dicts, output_dir):
        self.cfg = cfg
        self.val_dicts = val_dicts
        self.output_dir = output_dir
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
        os.makedirs(output_dir, exist_ok=True)
        
        # 분석용 데이터 저장소
        self.df_data = []

    def run_inference(self):
        print("🚀 Running Inference on All Data...")
        predictor = DefaultPredictor(self.cfg)
        
        for d in tqdm(self.val_dicts, desc="Inferencing"):
            img = cv2.imread(d["file_name"])
            if img is None: continue
            
            outputs = predictor(img)
            
            # GT와 매칭하여 데이터 수집
            self._process_single_image(d, outputs)
            
        # 데이터프레임 생성
        self.df = pd.DataFrame(self.df_data)
        print(f"📊 Collected {len(self.df)} matched samples.")

    def _process_single_image(self, input_record, outputs):
        pred = outputs["instances"].to("cpu")
        
        # GT 로드
        gt_boxes = []
        gt_classes = []
        gt_dists = []
        
        for ann in input_record["annotations"]:
            gt_boxes.append(ann["bbox"]) # XYWH_ABS
            gt_classes.append(ann["category_id"])
            gt_dists.append(ann["distance"])
            
        if not gt_boxes: return
        
        # XYWH -> XYXY 변환 (수동 변환이 더 확실함)
        gt_boxes_xyxy = []
        for box in gt_boxes:
            # [x, y, w, h] -> [x1, y1, x2, y2]
            gt_boxes_xyxy.append([box[0], box[1], box[0]+box[2], box[1]+box[3]])
            
        gt_boxes_tensor = torch.tensor(gt_boxes_xyxy, dtype=torch.float32)

        # 매칭 (IoU > 0.5)
        if len(pred) == 0: return
        
        # [핵심 수정] 텐서를 Boxes 객체로 감싸야 pairwise_iou가 작동함!
        # pred.pred_boxes는 이미 Boxes 객체임.
        ious = pairwise_iou(pred.pred_boxes, Boxes(gt_boxes_tensor))
        
        matched_vals, matched_idxs = ious.max(dim=1)
        
        for i, (val, idx) in enumerate(zip(matched_vals, matched_idxs)):
            if val > 0.5: # 매칭 성공
                gt_dist = gt_dists[idx.item()]
                
                # GT Distance가 유효할 때만 저장
                if gt_dist > 0:
                    # pred_distances가 1차원인지 확인 후 값 추출
                    if hasattr(pred, "pred_distances"):
                        # 텐서에서 값 하나만 뽑을 때는 .item() 사용
                        pred_dist = pred.pred_distances[i].item() # m 복원
                    else:
                        pred_dist = 0.0
                        
                    pred_cls = pred.pred_classes[i].item()
                    pred_score = pred.scores[i].item()
                    
                    # 클래스 이름 안전하게 가져오기
                    if pred_cls < len(self.class_names):
                        cls_name = self.class_names[pred_cls]
                    else:
                        cls_name = "unknown"
                    
                    self.df_data.append({
                        "Class_ID": pred_cls,
                        "Class_Name": cls_name,
                        "GT_Dist": gt_dist,
                        "Pred_Dist": pred_dist,
                        "Error": pred_dist - gt_dist,
                        "AbsError": abs(pred_dist - gt_dist),
                        "APE": abs(pred_dist - gt_dist) / (gt_dist + 1e-6) * 100.0,
                        "IoU": val.item(),
                        "Score": pred_score
                    })

    def analyze_distance(self):
        print("\n📏 [Distance Analysis] Calculating Metrics...")
        
        # 거리 구간 정의
        ranges = [
            (0, 10, "0-10m"), 
            (10, 30, "10-30m"), 
            (30, 50, "30-50m"), 
            (50, 100, "50-100m"),
            (0, 100, "Overall") # 전체 포함
        ]
        
        results = []

        # 1. 클래스별 x 구간별 분석
        for class_name in self.class_names:
            class_df = self.df[self.df["Class_Name"] == class_name]
            if len(class_df) == 0: continue
            
            for start, end, range_name in ranges:
                mask = (class_df["GT_Dist"] >= start) & (class_df["GT_Dist"] < end)
                sub_df = class_df[mask]
                
                if len(sub_df) > 0:
                    mae = sub_df["AbsError"].mean()
                    mape = sub_df["APE"].mean()
                    acc_10 = (sub_df["APE"] <= 10.0).mean() * 100
                    
                    results.append({
                        "Class": class_name,
                        "Range": range_name,
                        "Count": len(sub_df),
                        "MAE(m)": round(mae, 2),
                        "MAPE(%)": round(mape, 2),
                        "Acc_10(%)": round(acc_10, 2)
                    })
        
        # 2. 전체(All Classes) x 구간별 분석
        for start, end, range_name in ranges:
            mask = (self.df["GT_Dist"] >= start) & (self.df["GT_Dist"] < end)
            sub_df = self.df[mask]
            if len(sub_df) > 0:
                results.append({
                    "Class": "ALL_CLASSES",
                    "Range": range_name,
                    "Count": len(sub_df),
                    "MAE(m)": round(sub_df["AbsError"].mean(), 2),
                    "MAPE(%)": round(sub_df["APE"].mean(), 2),
                    "Acc_10(%)": round((sub_df["APE"] <= 10.0).mean() * 100, 2)
                })

        # CSV 저장
        res_df = pd.DataFrame(results)
        # 보기 좋게 정렬
        res_df = res_df.sort_values(by=["Class", "Range"])
        save_path = os.path.join(self.output_dir, "final_metrics_distance.csv")
        res_df.to_csv(save_path, index=False)
        print(f"✅ Distance Metrics saved to: {save_path}")
        print(res_df.head(10))

    def analyze_ap(self):
        print("\n🎯 [AP Analysis] Calculating mAP per Class & Range...")
        # Detectron2의 COCOEvaluator를 사용하여 AP 계산
        # 거리 구간별로 데이터셋을 필터링하여 반복 평가
        
        ranges = [
            (0, 100, "Overall"),
            (0, 10, "0-10m"), 
            (10, 30, "10-30m"), 
            (30, 50, "30-50m"), 
            (50, 100, "50-100m")
        ]
        
        ap_results = []
        
        # 모델 로드 (한 번만)
        predictor = DefaultPredictor(self.cfg)
        
        for start, end, range_name in ranges:
            print(f"   -> Evaluating AP for range: {range_name}...")
            
            # 해당 구간의 GT만 남긴 임시 데이터셋 생성
            temp_dataset_name = f"temp_eval_{range_name}"
            filtered_dicts = self._filter_dataset_by_range(start, end)
            
            if len(filtered_dicts) == 0:
                print(f"      (Skipping {range_name}: No Data)")
                continue
                
            if temp_dataset_name in DatasetCatalog.list(): DatasetCatalog.remove(temp_dataset_name)
            DatasetCatalog.register(temp_dataset_name, lambda: filtered_dicts)
            MetadataCatalog.get(temp_dataset_name).set(thing_classes=self.class_names)
            
            # Evaluator 실행
            evaluator = COCOEvaluator(temp_dataset_name, output_dir=self.output_dir)
            val_loader = build_detection_test_loader(self.cfg, temp_dataset_name, mapper=test_mapper)
            res = inference_on_dataset(predictor.model, val_loader, evaluator)
            
            # 결과 파싱 (bbox AP)
            bbox_res = res.get("bbox", {})
            # 전체 mAP
            ap_results.append({
                "Class": "ALL_CLASSES", "Range": range_name, 
                "AP": round(bbox_res.get("AP", 0), 2), "AP50": round(bbox_res.get("AP50", 0), 2)
            })
            
            # 클래스별 AP (COCOEvaluator는 기본적으로 클래스별 AP를 리턴하지 않으므로 로그 파싱 필요하지만,
            # 여기서는 편의상 전체 AP만 저장하거나, per-category AP를 지원하는 커스텀 evaluator 필요)
            # *Detectron2 기본 COCOEvaluator는 per-category AP를 '출력'은 하지만 리턴값에는 포함 안 시킬 수 있음.
            #  -> per_category_ap가 리턴되도록 설정되어 있는지 확인 필요. 보통은 안 됨.
            #  -> 따라서 여기서는 '전체 mAP'만 기록합니다. 클래스별 AP는 'Overall' 구간에서 로그를 확인하세요.

        # CSV 저장
        ap_df = pd.DataFrame(ap_results)
        save_path = os.path.join(self.output_dir, "final_metrics_ap.csv")
        ap_df.to_csv(save_path, index=False)
        print(f"✅ AP Metrics saved to: {save_path}")
        print(ap_df)

    def _filter_dataset_by_range(self, start, end):
        filtered = []
        for d in self.val_dicts:
            new_d = copy.deepcopy(d)
            new_anns = []
            for ann in new_d["annotations"]:
                dist = ann.get("distance", -1)
                
                # [추가] segmentation 필드 강제 보정 (없으면 bbox로 만듦)
                if "segmentation" not in ann:
                    x, y, w, h = ann["bbox"]
                    # 가짜 폴리곤 (사각형) 생성: [x,y, x+w,y, x+w,y+h, x,y+h]
                    ann["segmentation"] = [[x, y, x+w, y, x+w, y+h, x, y+h]]
                
                if dist > 0 and start <= dist < end:
                    new_anns.append(ann)
                elif start == 0 and end == 100: 
                    new_anns.append(ann)
            
            if len(new_anns) > 0:
                new_d["annotations"] = new_anns
                filtered.append(new_d)
        return filtered

# =========================================================
# [메인 실행]
# =========================================================
def main():
    print("🚀 Starting Final Detailed Evaluation...")
    
    # 1. 데이터셋 등록
    val_dicts, val_classes = get_val_dicts(TEST_JSON_PATH)
    DatasetCatalog.register("final_test_set", lambda: val_dicts)
    MetadataCatalog.get("final_test_set").set(thing_classes=val_classes)
    
    # 2. Config 설정
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NAME = "DistanceROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(val_classes)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.DATASETS.TEST = ("final_test_set",)
    
    # 3. 평가 실행
    evaluator = DeepEvaluator(cfg, val_dicts, OUTPUT_DIR)
    
    # (1) 추론 및 거리 오차 분석
    evaluator.run_inference()
    evaluator.analyze_distance()
    
    # (2) AP 분석 (시간이 좀 걸립니다)
    evaluator.analyze_ap()
    
    print(f"\n✨ 모든 평가가 완료되었습니다. '{OUTPUT_DIR}' 폴더를 확인하세요.")

if __name__ == "__main__":
    main()