# 📦 PCB Defect Detection & Real-Time Inspection System

## Overview
This project focuses on building a **high-speed AI-powered quality inspection system** for **Printed Circuit Boards (PCBs)** in a manufacturing environment.

The system processes images from a live camera feed and:
- Classifies each PCB as **Pass** or **Defect**
- Localizes defects using visual explanations
- Runs at **production-line speed (>10 FPS)**

---

## 🎯 Objectives
- Achieve **real-time defect detection**
- Ensure **high accuracy with low validation loss**
- Provide **model interpretability**
- Enable **deployment-ready inference**

---

## 📂 Dataset Structure
Dataset/
│
├── train/
│ ├── defect/ 
│ └── pass/
│
└── val/
├── defect/ 
└── pass/ 


---

## 🧠 Project Workflow

### 🔹 Task 1: Data Preparation
- Applied **real-time data augmentation** using `ImageDataGenerator`
- Techniques used:
  - Rotation
  - Zooming
  - Flipping
  - Brightness adjustments
- Visualized augmented batches to ensure:
  - Realism
  - Variability
  - No distortion of PCB features

---

### 🔹 Task 2: Core Modeling
- Implemented **Transfer Learning** using:
  - ResNet50 / MobileNetV2 (pre-trained models)
- Strategy:
  - Frozen base layers
  - Custom classification head
  - Fine-tuning for improved accuracy
- Focus:
  - Minimize **validation loss**
  - Prevent **overfitting** using learning curves

---

### 🔹 Task 3: Model Interpretability
- Implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Purpose:
  - Visualize model attention
  - Verify model focuses on **actual defect regions**
- Output:
  - Heatmaps overlaid on PCB images

---

### 🔹 Task 4: Inference Optimization & Live Demo
- Built a **real-time inference pipeline** using:
  - OpenCV (webcam simulation)
  - Frame-by-frame prediction
- Optimized model for:
  - High throughput (>10 FPS)
  - Efficient deployment (`.h5` / SavedModel)

#### 🔥 Improvement:
- Integrated **YOLOv5** for enhanced:
  - Object detection
  - Defect localization
  - Real-time performance

---

## ⚙️ Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- YOLOv5
- NumPy / Matplotlib

---

#LArge files,yolov5 folder
due to size limitations,this folder is hosted on google drive:
https://drive.google.com/drive/folders/1tMkD30DMYiKF0-wejuHDm6EZgWsSxsWx?usp=drive_link

#LArge files,P_res_50.h5
due to size limitations,this folder is hosted on google drive:
https://drive.google.com/file/d/1YF_qdIOG8WbMsQ94QBY0suYbVdIBhFt2/view?usp=sharing

## 📊 Features

✅ Real-time PCB inspection  
✅ Data augmentation for robustness  
✅ Transfer learning for high accuracy  
✅ Grad-CAM for explainability  
✅ YOLOv5 for defect localization  
✅ Live webcam inference  

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Le68vi/PCB.git
cd pcb-defect-detection

## Install dependencies
pip install -r requirements.txt


#Train the model
python res.py

# Run real-time inference
python speed.py

📈# Performance Goals
Accuracy: High classification performance
Speed: >10 FPS inference
Robustness: Handles lighting & orientation variations

🔍 #Future Improvements
Deploy using TensorRT / ONNX for faster inference
Edge deployment on embedded devices
Expand dataset for better generalization
Multi-class defect detection
🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

📜 License

This project is licensed under the MIT License.

👤 Author

MUKUL Kandwal
VAIBHAV Gupta
AYUSH Gaudani
SAMIR Shaikh 
