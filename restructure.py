import os
import shutil
from pathlib import Path

def final_cleanup():
    base_dir = Path.cwd()
    print(f"최종 정리 시작: {base_dir}")

    # 1. 로그 파일 삭제
    log_files = [
        "src/2_training/nohup.out",
        "src/2_training/output.log",
        "src/2_training/train_log.out",
        "src/3_evaluation/output_se.log"
    ]
    for log_path in log_files:
        p = base_dir / log_path
        if p.exists():
            p.unlink()
            print(f"🗑️ 로그 파일 삭제: {log_path}")

    # 2. 임시 lock 파일 삭제
    for p in base_dir.rglob("*.lock"):
        if p.is_file():
            p.unlink()
            print(f"🗑️ 임시 파일 삭제: {p.relative_to(base_dir)}")

    # 3. 모델 가중치 이동
    weight_src = base_dir / "src/3_evaluation/clear/model_best_dist.pth"
    weight_dest = base_dir / "weights/model_best_dist.pth"
    if weight_src.exists():
        shutil.move(str(weight_src), str(weight_dest))
        print(f"🚚 가중치 이동: {weight_src.relative_to(base_dir)} -> weights/")

    # 4. 설정 파일(yaml) 이동
    yaml_src = base_dir / "src/3_evaluation/clear/config.yaml"
    yaml_dest = base_dir / "configs/config.yaml"
    if yaml_src.exists():
        shutil.move(str(yaml_src), str(yaml_dest))
        print(f"🚚 설정 파일 이동: {yaml_src.relative_to(base_dir)} -> configs/")

    print("\n✅ 최종 정리 완료.")

if __name__ == "__main__":
    final_cleanup()