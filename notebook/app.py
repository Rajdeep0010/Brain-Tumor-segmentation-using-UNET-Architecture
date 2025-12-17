import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image

#Prediction function
def predict_and_explain(model, image_array, layer_name, img_size=128):
  #load and preprocess the image
    img_resized = cv2.resize(image_array, (img_size, img_size))
    img_norm = img_resized / 255.0
    input_img = np.expand_dims(img_norm, axis=0)

    #prediction of the mask
    pred_mask = model.predict(input_img)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    pred_mask_vis = np.squeeze(pred_mask)

    #Integrating GRADE cam
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_img)
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(pooled_grads):
        cam += w * conv_outputs[:, :, i]
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    #applying heatmap overlay
    cam_resized = cv2.resize(cam, (img_size, img_size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * heatmap + 0.4 * (img_resized * 255))

    return img_resized, pred_mask_vis, overlay



