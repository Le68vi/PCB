# PCB Prediction with Grad-CAM (Offline Images)

import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model


# LOAD TRAINED MODEL
model_path = "P_res_50.h5"   #trained model
model = load_model(model_path)
print(f"Loaded model: {model_path}")


# CLASS LABELS
class_names = ["Defect", "Pass"]  # index 0=Defect, 1=Pass

# IMAGE PREPROCESSING
def preprocess(img):
    img_resized = cv2.resize(img, (224, 224))
    img_array = img_resized.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# PREDICTION FUNCTION
def infer(img):
    input_data = preprocess(img)
    output = model.predict(input_data, verbose=0)
    predicted_index = np.argmax(output)
    confidence = float(np.max(output))
    label = class_names[predicted_index]
    return label, confidence, input_data

# GRAD-CAM FUNCTIONS
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return superimposed_img


# TEST FOLDER SETUP
test_folder = "opp"  # Folder containing PCB images
if not os.path.exists(test_folder):
    print(f"Error: {test_folder} does not exist.")
    exit()

# PROCESS ALL IMAGES
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read {img_name}, skipping.")
        continue

    # Prediction
    label, confidence, input_data = infer(img)

    # Grad-CAM overlay
    heatmap = make_gradcam_heatmap(input_data, model)
    img_with_heatmap = overlay_heatmap(img, heatmap)

    # Display results
    print(f"{img_name}: {label} ({confidence:.2f})")
    color = (0, 255, 0) if label == "Pass" else (0, 0, 255)
    cv2.putText(img_with_heatmap, f"{label} ({confidence:.2f})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("PCB Prediction with Grad-CAM", img_with_heatmap)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("All images processed with Grad-CAM.")