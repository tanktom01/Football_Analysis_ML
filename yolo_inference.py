from ultralytics import YOLO

model = YOLO("models/yolo11m")

results = model.predict("input_videos/test_video", save=True)
print(results[0])
print("=======================")

for box in results[0].boxes:
    print(box)
