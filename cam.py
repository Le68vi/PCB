import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Load new model
model = load_model("P_res_50.h5")

# Image Preprocessing
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Grad-CAM Function
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
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Overlay Heatmap
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


# Process Validation Folder
input_folder = "Dataset/val"    # dataset path
output_folder = "gradcam_results"
os.makedirs(output_folder, exist_ok=True)

for class_name in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_name)
    if not os.path.isdir(class_path):
        continue
    output_class_path = os.path.join(output_folder, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    # Optional: limit for demo
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img_array = preprocess_image(img_path)
            heatmap = make_gradcam_heatmap(img_array, model)
            result = overlay_heatmap(img_path, heatmap)
            cv2.imwrite(os.path.join(output_class_path, img_name), result)
            print(f"Processed: {img_name}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

print("Grad-CAM generation completed!")