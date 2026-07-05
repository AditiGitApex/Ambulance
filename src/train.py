"""
train.py
Train YOLOv8n on merged ambulance + non-ambulance dataset.
"""

from ultralytics import YOLO
from pathlib import Path


def main():
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / "merged_dataset" / "data.yaml"
    models_dir = project_root / "models"

    print(f"Dataset config: {data_yaml}")
    print(f"Models output:  {models_dir}\n")

    # Pre-trained YOLOv8n base
    model = YOLO('yolov8n.pt')

    # Training
    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=416,
        batch=16,
        device='cpu',           # GPU ho toh 0 likh dena

        # Output
        project=str(models_dir),
        name='ambulance_v5',
        exist_ok=False,

        # Optimization
        patience=15,            # 15 epochs improvement nahi → early stop
        save=True,
        save_period=5,          # checkpoint har 5 epochs
        cache=True,             # RAM caching → CPU speed boost
        workers=4,

        # Augmentations (Roboflow se nahi liye, yahan apply karenge)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        translate=0.1,
        scale=0.5,

        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)

    best_weights = models_dir / 'ambulance_v5' / 'weights' / 'best.pt'
    print(f"Best weights: {best_weights}")

    # Final validation
    print("\nRunning validation on best weights...")
    metrics = model.val()

    print(f"\nOverall mAP50:    {metrics.box.map50:.3f}")
    print(f"Overall mAP50-95: {metrics.box.map:.3f}")

    print("\nPer-class mAP50-95:")
    for i, name in metrics.names.items():
        try:
            print(f"  {name:20s}: {metrics.box.maps[i]:.3f}")
        except (IndexError, AttributeError):
            pass


if __name__ == "__main__":
    main()