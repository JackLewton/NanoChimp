from ultralytics import YOLO

# 1. load model, change path to point to your weights
model = YOLO('runs/detect/yolo_training/gpu_bounding_box_model6/weights/best.pt')

# 2. eval (default set to val split)
metrics = model.val(data = "yolo_dataset/data.yaml")

# 3. print eval metrics
print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")

precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1_score:.4f}")
