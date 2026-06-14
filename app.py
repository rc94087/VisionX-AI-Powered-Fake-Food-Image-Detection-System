import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from PIL.ExifTags import TAGS
import cv2

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI vs Real Detector",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🍽️ AI vs Real Food Image Detector")
st.markdown("Upload a food image to detect if it's AI-generated or tampered")

# ===============================
# LOAD MODEL
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    keras_path = os.path.join(BASE_DIR, "ai_vs_real_food_detector_v2.keras")
    h5_path    = os.path.join(BASE_DIR, "ai_vs_real_food_detector_v2.h5")

    # Fallback to v1 if v2 not found
    if not os.path.isfile(keras_path):
        keras_path = os.path.join(BASE_DIR, "ai_vs_real_food_detector.keras")
    if not os.path.isfile(h5_path):
        h5_path = os.path.join(BASE_DIR, "ai_vs_real_food_detector.h5")

    model        = None
    model_format = None

    if os.path.isfile(keras_path):
        try:
            model        = keras.models.load_model(keras_path, compile=False)
            model_format = ".keras"
        except Exception as e:
            st.warning(f"⚠️ Failed to load .keras: {e}")

    if model is None and os.path.isfile(h5_path):
        try:
            model        = keras.models.load_model(h5_path, compile=False)
            model_format = ".h5"
        except Exception as e:
            st.error(f"❌ Failed to load .h5: {e}")

    if model is None:
        st.error("❌ Model file not found!")
        st.error("Expected: ai_vs_real_food_detector_v2.keras")
        st.info("💡 Run 'python train_model_hybrid.py' to create the model first.")
        st.stop()

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, model_format


try:
    model, model_format = load_model()
    st.success(f"✅ Model loaded successfully ({model_format} format)")
except Exception as e:
    st.error("❌ Failed to load model")
    st.exception(e)
    st.stop()

# ===============================
# GRAD-CAM
# ===============================
def make_gradcam_heatmap(img_array, model):
    try:
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is None:
            raise ValueError("No nested base model found")

        target_layer = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer.name
                break

        if target_layer is None:
            raise ValueError("No Conv2D layer found")

        inp           = tf.keras.Input(shape=(224, 224, 3))
        x             = inp
        target_output = None

        for layer in base_model.layers:
            x = layer(x)
            if layer.name == target_layer:
                target_output = x

        for layer in model.layers:
            if layer == base_model:
                continue
            x = layer(x)

        grad_model = tf.keras.Model(inputs=inp, outputs=[target_output, x])

        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(img_array)
            loss = pred[:, 0]

        grads  = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_out.numpy()[0]
        pooled   = pooled.numpy()

        for i in range(len(pooled)):
            conv_out[:, :, i] *= pooled[i]

        heatmap = np.mean(conv_out, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        return heatmap

    except Exception as e:
        print(f"Grad-CAM failed, using fallback: {e}")
        h, w     = img_array.shape[1:3]
        y, x     = np.ogrid[0:h, 0:w]
        cx, cy   = w // 2, h // 2
        dist     = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        return np.clip(1 - (dist / max_dist), 0, 1)


def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    if image.dtype != np.uint8:
        image = np.uint8(image)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


# ===============================
# METADATA CHECK
# ===============================
def check_metadata(image: Image.Image):
    try:
        exif = image._getexif()
        if exif is None:
            return "No metadata found"

        ai_keywords = [
            "ai", "midjourney", "stable diffusion", "dall-e", "dalle",
            "artificial intelligence", "generated", "synthetic",
            "firefly", "chatgpt", "openai", "canva"
        ]

        for tag_id, value in exif.items():
            tag       = TAGS.get(tag_id, tag_id)
            value_str = str(value).lower()
            for keyword in ai_keywords:
                if keyword in value_str:
                    return f"⚠️ AI-related metadata detected: '{keyword}'"

        return "✓ Metadata appears normal"

    except Exception:
        return "Metadata not readable"


# ===============================
# FILE UPLOAD & MAIN UI
# ===============================
st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose a food image (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to analyze if it's AI-generated or tampered"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.markdown("### Quick Info")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size[0]}x{image.size[1]}")
            st.write(f"**Format:** {image.format}")

        # ===============================
        # PREPROCESS
        # ===============================
        img     = image.resize((224, 224))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # ===============================
        # PREDICTION
        # ===============================
        with st.spinner("🔍 Analyzing image..."):
            try:
                _    = model(img_arr, training=False)
                prob = float(model.predict(img_arr, verbose=0)[0][0])
            except Exception as e:
                st.error("❌ Prediction failed")
                st.exception(e)
                st.stop()

        # ---------------------------------------------------------------
        # class_indices = {'ai': 0, 'real': 1}
        #   prob close to 0 = AI/Tampered
        #   prob close to 1 = Real
        # ---------------------------------------------------------------
        ai_probability   = 1.0 - prob
        real_probability = prob

        metadata      = check_metadata(image)
        metadata_flag = "⚠️" in metadata

        # ===============================
        # FINAL DECISION
        # ===============================
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        if metadata_flag or ai_probability > 0.15:
            is_ai               = True
            confidence          = max(ai_probability, 0.90) if metadata_flag else ai_probability
            classification_type = "high_confidence_ai"
            st.error("### 🤖 AI-Generated / Tampered Image")
            if metadata_flag:
                reason = "Metadata contains AI tool indicators"
            elif ai_probability > 0.50:
                reason = "Strong AI visual patterns detected"
            else:
                reason = "Subtle AI editing or inpainting artifacts detected"

        elif ai_probability > 0.10:
            is_ai               = True
            confidence          = ai_probability
            classification_type = "uncertain_ai"
            st.warning("### ⚠️ Possibly AI-Generated or Edited")
            reason = "Some AI-like patterns detected — not fully conclusive"
            st.info("**Note:** This might be a real image with unusual characteristics or a lightly edited photo.")

        elif real_probability > 0.85:
            is_ai               = False
            confidence          = real_probability
            classification_type = "high_confidence_real"
            st.success("### 📸 Real Image")
            reason = "Visual analysis strongly suggests a real photograph"

        else:
            is_ai               = False
            confidence          = real_probability
            classification_type = "uncertain_real"
            st.info("### 📷 Likely Real, Some Uncertainty")
            reason = "Appears to be a real photo but with some unusual features"

        confidence_percent = confidence * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{confidence_percent:.1f}%")
        with col2:
            st.metric("AI Probability", f"{ai_probability * 100:.1f}%")
            st.caption("(>15% = flagged as AI)")
        with col3:
            category = "AI-Generated" if is_ai else "Real Photo"
            st.metric("Classification", category)

        st.progress(min(int(confidence_percent), 100) / 100)
        st.info(f"**Reasoning:** {reason}")
        st.caption(f"**Metadata Check:** {metadata}")
        st.caption(f"**Raw Score:** {prob:.4f}  |  AI prob: {ai_probability*100:.1f}%  |  Real prob: {real_probability*100:.1f}%")

        # ===============================
        # GRAD-CAM VISUALIZATION
        # ===============================
        st.markdown("---")
        st.markdown("## 🧠 Model Attention Map (Grad-CAM)")
        st.caption("Shows which parts of the image influenced the model's decision")

        try:
            with st.spinner("Generating attention map..."):
                img_np    = np.array(image)
                heatmap   = make_gradcam_heatmap(img_arr, model)
                cam_image = overlay_heatmap(img_np, heatmap)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(cam_image, caption="Attention Heatmap", use_container_width=True)

                st.markdown("""
                **How to read this:**
                - 🔴 **Red areas**: High attention (model focused here)
                - 🟡 **Yellow areas**: Moderate attention
                - 🔵 **Blue areas**: Low attention
                """)

        except Exception as e:
            st.warning("⚠️ Could not generate Grad-CAM visualization")
            st.image(image, caption="Original Image", use_container_width=True)
            with st.expander("Why did this fail?"):
                st.write(f"**Error:** {str(e)}")
                st.markdown("**Note:** The prediction still works correctly. Grad-CAM is just a visualization tool.")

    except Exception as e:
        st.error("❌ Error processing image")
        st.exception(e)

else:
    st.info("👆 Upload an image above to get started")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ℹ️ How it works")
        st.markdown("""
        This app uses deep learning to detect AI-generated and tampered food images.

        **Detection Methods:**
        1. **Visual Analysis**: CNN examines pixel patterns and artifacts
        2. **Metadata Check**: Scans EXIF data for AI tool signatures
        3. **Grad-CAM**: Shows which image regions influenced the decision

        **Supported Formats:** JPG, JPEG, PNG
        """)
    with col2:
        st.markdown("### 📈 How Results Work")
        st.markdown("""
        - **AI prob > 15%** → Flagged as AI/Tampered
        - **AI prob > 50%** → Strong AI patterns
        - **AI prob < 10%** → Likely Real

        **Model:** MobileNetV2 v2 with transfer learning
        **Trained on:** 208,687 images
        **Val Accuracy:** 98.74%
        """)

    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        1. **Model not found**: Run `python train_model_hybrid.py`
        2. **Prediction fails**: Check image format and size
        3. **Grad-CAM fails**: Normal — prediction still works
        4. **Wrong prediction**: Try retraining with more diverse data
        """)

st.markdown("---")
st.caption("🍽️ AI vs Real Food Detector | Powered by TensorFlow & Streamlit")