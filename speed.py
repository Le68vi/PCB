import os
import time
import cv2
import torch

#Load your trained YOLOv5 model
model_path = r"yolov5\yolov5s.pt"

# Load YOLOv5 custom model
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local', force_reload=True)
model.conf = 0.25  # confidence threshold
model.iou = 0.45   # NMS IoU threshold

print("✅ YOLOv5 model loaded!")

#Folder of PCB images
image_folder = r"C:\Users\mukul\OneDrive\Desktop\man\P1\code\PCB-DEFECT-\opp"
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")])

# Simulate live feed speed (~10 FPS)
simulate_delay = 0.1

#Loop through images
for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Could not load {img_file}")
        continue

    # Run YOLOv5 inference
    results = model(img)

    # Extract detections: x1, y1, x2, y2, confidence, class_id
    detections = results.xyxy[0].cpu().numpy()

    # Determine PASS / DEFECT
    if len(detections) > 0:
        verdict = "DEFECT"
    else:
        verdict = "PASS"

    # Print results
    print(f"Image: {img_file} --> {verdict}")

    # Draw bounding boxes and class labels (optional for debugging)
    for *box, conf, cls_id in detections:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if verdict == "DEFECT" else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Overlay verdict text on image
    cv2.putText(img, verdict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255) if verdict == "DEFECT" else (0, 255, 0), 2)

    # Show annotated image
    cv2.imshow("YOLOv5 PCB Inspection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

    # Simulate live feed
    time.sleep(simulate_delay)

cv2.destroyAllWindows()