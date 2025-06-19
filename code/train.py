from ultralytics import YOLO
import os
import cv2
import numpy as np
from thop import profile
import torch
import time
import csv
from itertools import product

# 여러 기법을 사용하여 학습 및 예측을 수행하는 코드
# === 공통 기반 설정 ===
base_config = {
    "batch_size": 16,
    "lrf": 0.01
}

# === 조합할 항목들 ===
models = ["yolo11n-seg.pt"]
optimizers = ["SGD", "AdamW"]
epochs_list = [100, 1000]
imgsz_list = [640, 768, 896, 1024]
lr0_list = [0.01, 0.001]

# === 조합 생성 ===
experiments = []

for model, optimizer, epochs, imgsz, lr0 in product(models, optimizers, epochs_list, imgsz_list, lr0_list): 
    # 저장 폴더를 위해 run_name 생성
    model_base = model.replace(".pt", "")
    run_name = f"strawberry_{model_base}_{optimizer.lower()}"
    if epochs != 100:
        run_name += f"_e{epochs}"
    if imgsz != 640:
        run_name += f"_img{imgsz}"
    if lr0 != 0.01:
        run_name += f"_lr{lr0}"

    config = {
        "model_name": model,
        "run_name": run_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": base_config["batch_size"],
        "lr0": lr0,
        "lrf": base_config["lrf"],
        "optimizer": optimizer
    }

    experiments.append(config)

for i, exp in enumerate(experiments):
    # === [0] 저장 경로 및 모델 경로 확인 ===
    project_dir = "/home/cv_task/Seungun/strawberry/nbt/original/outputs_3w"
    run_name = exp["run_name"]
    save_dir = os.path.join(project_dir, run_name)
    best_model_path = os.path.join(save_dir, "weights", "best.pt")

    # ❌ 이미 학습한 경우 스킵
    if os.path.exists(best_model_path):
        print(f"⏭️  이미 학습된 모델: {run_name}, 건너뜁니다.")
        continue

    # === [1] 모델 초기화 및 학습 ===
    model = YOLO(exp["model_name"])

    results = model.train(
        data="/home/cv_task/Seungun/strawberry/nbt/original/data_3w.yaml",
        epochs=exp["epochs"], imgsz=exp["imgsz"], batch=exp["batch_size"],
        lr0=exp["lr0"], lrf=exp["lrf"], optimizer=exp["optimizer"],
        amp=True, workers=0,
        verbose=False,
        project=project_dir, name=run_name, exist_ok=True
    )

    # === [2] 학습된 모델 로드 ===
    best_model_path = os.path.join(save_dir, "weights", "best.pt")
    model = YOLO(best_model_path)

    # === [3] 검증 이미지 예측 시각화 및 저장 ===
    image_dir = "/home/cv_task/Seungun/strawberry/nbt/original/data_3w/images/val"
    label_dir = "/home/cv_task/Seungun/strawberry/nbt/original/data_3w/labels/val"
    output_dir = os.path.join(save_dir, "val_predictions")
    os.makedirs(output_dir, exist_ok=True)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", "JPG"))])[:10]

    for image_name in images:
        img_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")
        save_path = os.path.join(output_dir, image_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 이미지 로드 실패: {img_path}")
            continue
        h, w = img.shape[:2]

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7 and len(parts[1:]) % 2 == 0:
                        cls = int(parts[0])
                        points = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
                        points[:, 0] *= w
                        points[:, 1] *= h
                        pts_int = np.round(points).astype(np.int32)
                        color = colors[cls % len(colors)]
                        cv2.polylines(img, [pts_int], isClosed=True, color=color, thickness=2)
                        cv2.putText(img, f"GT:{cls}", tuple(pts_int[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        r = model(img_path)[0]
        if r.masks is not None and r.masks.data is not None:
            for i, mask in enumerate(r.masks.data):
                cls = int(r.boxes.cls[i].item())
                conf = r.boxes.conf[i].item()
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
                if contours and len(contours[0]) > 0:
                    x, y, w_box, h_box = cv2.boundingRect(contours[0])
                    cv2.putText(img, f"P:{cls} {conf:.2f}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(save_path, img)
        print(f"✅ 예측 결과 저장: {save_path}")

    # === [4] 모델 리소스 측정 및 저장 ===
    print("\n📊 모델 리소스 측정 중...")
    dummy_input = torch.randn(1, 3, 640, 640).to(model.device)
    model.model.to(model.device)

    flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)

    torch.cuda.synchronize()
    start = time.time()
    _ = model.model(dummy_input)
    torch.cuda.synchronize()
    end = time.time()

    inference_time = (end - start) * 1000  # ms
    fps = 1000 / inference_time

    csv_path = os.path.join(save_dir, "model_resource.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Params(M)", "FLOPs(G)", "Inference time(ms)", "FPS"])
        writer.writerow([
            "yolo11m-seg",
            f"{params / 1e6:.2f}",
            f"{flops / 1e9:.2f}",
            f"{inference_time:.2f}",
            f"{fps:.2f}"
        ])

    print(f"\n✅ 학습 완료 및 예측 결과 저장 완료")
    print(f"📁 모델 저장 위치: {best_model_path}")
    print(f"🖼️ 예측 이미지 저장 위치: {output_dir}")
    print(f"📄 리소스 저장 위치: {csv_path}")
    
    # 메모리 정리
    torch.cuda.empty_cache()