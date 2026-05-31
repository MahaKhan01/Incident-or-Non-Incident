
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import base64
from pathlib import Path
import gdown

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

# =========================
# Model and app configuration
# =========================


MODEL_PATH = Path("models/best_incident_model.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    url = "https://drive.google.com/file/d/18XYEm2JxVKv2dd7EnUBRiBPS3G36Kuw_/view?usp=drive_link"
    gdown.download(url, str(MODEL_PATH), quiet=False)

MODEL_CANDIDATES = [
    Path("models/best_incident_model.keras"),
    Path("best_incident_model.keras"),
    Path("/content/models/best_incident_model.keras"),
    Path("/content/best_incident_model.keras"),
]

DEFAULT_IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="AI Incident Classification",
    page_icon="🚨",
    layout="centered"
)

# =========================
# Helper functions
# =========================

def find_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return str(path)
    return None


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


@st.cache_resource
def load_model():
    model_path = find_model_path()
    if model_path is None:
        st.error(
            "Model file not found. Please place best_incident_model.keras in "
            "models/best_incident_model.keras before running the app."
        )
        st.stop()

    custom_objects = {
        "mobilenet_preprocess_input": mobilenet_preprocess_input,
        "resnet_preprocess_input": resnet_preprocess_input,
        "preprocess_input": resnet_preprocess_input,
    }

    try:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except TypeError:
        # Some TensorFlow/Keras versions need safe_mode=False for Lambda preprocessing layers.
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            safe_mode=False
        )


model = load_model()


def get_model_image_size(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]

        height = shape[1]
        width = shape[2]

        if height is not None and width is not None:
            return (int(width), int(height))
    except Exception:
        pass

    return DEFAULT_IMG_SIZE


IMG_SIZE = get_model_image_size(model)


# =========================
# Logo helper
# =========================

logo_base64 = None
for logo_path in [Path("logo.png"), Path("app/logo.png"), Path("/content/logo.png")]:
    if logo_path.exists():
        logo_base64 = image_to_base64(logo_path)
        break

if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="Project logo">'
else:
    logo_html = '<div class="logo-placeholder">🚨</div>'


# =========================
# Page styling and banner
# =========================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #f4f8fc;
    }}

    .block-container {{
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }}

    .top-banner {{
        background-color: #003B5C;
        padding: 22px 35px;
        border-radius: 0 0 14px 14px;
        display: flex;
        align-items: center;
        gap: 22px;
        margin-bottom: 30px;
        box-shadow: 0 4px 14px rgba(0, 59, 92, 0.18);
    }}

    .top-banner img {{
        width: 90px;
        height: auto;
    }}

    .logo-placeholder {{
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: #E8F2FF;
        color: #003B5C;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 46px;
        font-weight: 800;
    }}

    .top-banner-title {{
        color: white;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.2;
    }}

    .top-banner-subtitle {{
        color: #d6ecff;
        font-size: 16px;
        margin-top: 6px;
    }}

    h1, h2, h3 {{
        color: #003B5C;
    }}

    .intro-card {{
        background-color: #E8F2FF;
        border-left: 6px solid #0072CE;
        padding: 18px 20px;
        border-radius: 10px;
        margin-bottom: 22px;
        color: #102A43;
        font-size: 16px;
        box-shadow: 0 3px 10px rgba(0, 59, 115, 0.06);
    }}

    .warning-card {{
        background-color: #FFF4D6;
        border-left: 6px solid #F5A400;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 25px;
        color: #4A3500;
        font-size: 16px;
    }}

    .section-card {{
        background-color: #ffffff;
        border: 1px solid #d6e3f0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 22px;
        box-shadow: 0 3px 10px rgba(0, 59, 115, 0.07);
    }}

    [data-testid="stFileUploader"] {{
        background-color: #ffffff;
        border: 2px dashed #0072CE;
        border-radius: 12px;
        padding: 15px;
    }}

    .result-box {{
        background-color: #ffffff;
        border: 1px solid #c9d8e8;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 18px;
        box-shadow: 0 3px 10px rgba(0, 59, 115, 0.08);
    }}

    .incident {{
        color: #b00020;
        font-size: 26px;
        font-weight: 800;
    }}

    .non-incident {{
        color: #00703C;
        font-size: 26px;
        font-weight: 800;
    }}

    .footer {{
        text-align: center;
        color: #52616B;
        font-size: 13px;
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid #c9d8e8;
    }}

    hr {{
        border: none;
        border-top: 2px solid #c9d8e8;
        margin-top: 30px;
        margin-bottom: 25px;
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }}

        .top-banner {{
            flex-direction: column;
            text-align: center;
            padding: 20px 18px;
            gap: 12px;
        }}

        .top-banner img,
        .logo-placeholder {{
            width: 70px;
            height: 70px;
            font-size: 36px;
        }}

        .top-banner-title {{
            font-size: 23px;
        }}

        .top-banner-subtitle {{
            font-size: 14px;
        }}

        .intro-card,
        .warning-card,
        .result-box,
        .section-card {{
            padding: 16px;
            font-size: 15px;
        }}

        .incident,
        .non-incident {{
            font-size: 22px;
        }}
    }}
    </style>

    <div class="top-banner">
        {logo_html}
        <div>
            <div class="top-banner-title">AI-Based Incident Image and Video Classification</div>
            <div class="top-banner-subtitle">Community-submitted media classification prototype</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Intro and emergency notice
# =========================

st.markdown(
    """
    <div class="intro-card">
        This prototype classifies uploaded images or short videos as
        <b>Incident</b> or <b>Non-Incident</b> and provides a confidence score.
    </div>
    """,
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown("## ⚠️ Is it an emergency?")
    st.markdown(
        """
        **Call 999 now** if:

        - there is an immediate danger to life
        - someone is using violence or threatening to be violent
        - a crime is happening right now
        - the suspect is still at the scene

        ### Hearing or speech impairments

        - If you have pre-registered with the emergencySMS service, use your textphone service 18000 or text 999.
        - Call 999 BSL to use a British Sign Language interpreter.

        **Academic prototype notice:** This system does not contact the police or emergency services.

        It is only a university project demonstrating how AI could support classification of uploaded incident images and videos.
        """
    )

st.markdown(
    """
    <div class="warning-card">
        <b>Decision-support notice:</b> This system is a decision-support academic prototype only.
        It must not replace human judgement, official investigation, or emergency response.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Prediction function
# =========================

def predict_image(img):
    img = img.convert("RGB")
    img_resized = img.resize(IMG_SIZE)

    img_array = np.array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model.predict(img_array, verbose=0)[0][0])

    if prob >= 0.5:
        pred_label = "Incident"
        confidence = prob
    else:
        pred_label = "Non-Incident"
        confidence = 1 - prob

    return pred_label, float(confidence), float(prob)


# =========================
# Grad-CAM helpers
# =========================

def safe_call_layer(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def find_conv_base_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            conv_layers = [
                inner_layer for inner_layer in layer.layers
                if isinstance(inner_layer, tf.keras.layers.Conv2D)
            ]
            if len(conv_layers) > 0:
                return layer
    return None


def find_last_conv_layer(model_or_base):
    for layer in reversed(model_or_base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None


def generate_gradcam(img):
    img = img.convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    base_model = find_conv_base_model(model)

    if base_model is not None:
        last_conv_layer = find_last_conv_layer(base_model)
        if last_conv_layer is None:
            raise ValueError("No convolutional layer found inside the selected base model.")

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

        x = tf.convert_to_tensor(img_array, dtype=tf.float32)

        for layer in model.layers:
            if layer == base_model:
                break
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = safe_call_layer(layer, x)

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = grad_model(x, training=False)

            y = base_outputs
            reached_base_model = False

            for layer in model.layers:
                if layer == base_model:
                    reached_base_model = True
                    continue
                if reached_base_model:
                    if isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    y = safe_call_layer(layer, y)

            predicted_probability = y[:, 0]
            class_score = predicted_probability if predicted_probability[0] >= 0.5 else 1.0 - predicted_probability

    else:
        last_conv_layer = find_last_conv_layer(model)
        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the loaded model.")

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        x = tf.convert_to_tensor(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x, training=False)
            predicted_probability = predictions[:, 0]
            class_score = predicted_probability if predicted_probability[0] >= 0.5 else 1.0 - predicted_probability

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise ValueError("Gradients could not be calculated for Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if max_value != 0:
        heatmap = heatmap / max_value

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(img_resized)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    superimposed_img = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img


# =========================
# Video classification
# =========================

def classify_video(video_path, frame_interval=30, max_frames=60, max_display_frames=6):
    cap = cv2.VideoCapture(video_path)

    probs = []
    sampled_frames = []
    sampled_gradcams = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, IMG_SIZE)

            frame_array = np.expand_dims(frame_resized.astype("float32"), axis=0)
            prob = float(model.predict(frame_array, verbose=0)[0][0])
            probs.append(prob)

            if len(sampled_frames) < max_display_frames:
                sampled_frames.append((frame_resized, prob))

                try:
                    frame_pil = Image.fromarray(frame_resized)
                    gradcam_frame = generate_gradcam(frame_pil)
                    sampled_gradcams.append((gradcam_frame, prob))
                except Exception:
                    sampled_gradcams.append((None, prob))

            if len(probs) >= max_frames:
                break

        frame_count += 1

    cap.release()

    if len(probs) == 0:
        return None, None, None, 0, [], []

    avg_prob = float(np.mean(probs))

    if avg_prob >= 0.5:
        pred_label = "Incident"
        confidence = avg_prob
    else:
        pred_label = "Non-Incident"
        confidence = 1 - avg_prob

    return pred_label, confidence, avg_prob, len(probs), sampled_frames, sampled_gradcams


# =========================
# Upload and display
# =========================

st.markdown("### 📤 1. Upload Media")
st.info("Upload a JPG, PNG, MP4, AVI, or MOV file. The system will classify it as Incident or Non-Incident.")

uploaded_file = st.file_uploader(
    "Choose an image or short video file",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        img = Image.open(uploaded_file).convert("RGB")

        st.markdown("### 🖼️ 2. Uploaded Image Preview")

        preview_col1, preview_col2 = st.columns([1, 1])

        with preview_col1:
            st.image(
                img,
                caption="Uploaded image preview",
                use_container_width=True
            )

        with preview_col2:
            st.markdown(
                """
                <div class="section-card">
                    <b>File type:</b> Image<br>
                    <b>Analysis:</b> Incident classification<br>
                    <b>Output:</b> Prediction, confidence score, and Grad-CAM explanation
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.spinner("Analysing uploaded image..."):
            pred_label, confidence, raw_score = predict_image(img)

        label_class = "incident" if pred_label == "Incident" else "non-incident"

        st.markdown("### 📊 3. Classification Result")

        st.markdown(
            f"""
            <div class="result-box">
                <p>Predicted Class:</p>
                <p class="{label_class}">{pred_label}</p>
                <p><b>Confidence:</b> {confidence * 100:.2f}%</p>
                <p><b>Raw model score:</b> {raw_score:.4f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(confidence))

        st.markdown("### 🧠 4. Grad-CAM Explanation")

        try:
            with st.spinner("Generating Grad-CAM explanation..."):
                gradcam_img = generate_gradcam(img)

            st.image(
                gradcam_img,
                caption="Highlighted regions influenced the AI prediction.",
                use_container_width=True
            )

        except Exception as e:
            st.error("Grad-CAM could not be generated for this model.")
            st.write(str(e))

    elif "video" in file_type:
        st.markdown("### 🎞️ 2. Uploaded Video Preview")

        suffix = "." + uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name

        st.video(temp_path)

        with st.spinner("Extracting frames and analysing video..."):
            pred_label, confidence, avg_score, frames, sampled_frames, sampled_gradcams = classify_video(temp_path)

        if frames == 0:
            st.error("No frames could be extracted from the video.")
        else:
            label_class = "incident" if pred_label == "Incident" else "non-incident"

            st.markdown("### 📊 3. Video Classification Result")

            st.markdown(
                f"""
                <div class="result-box">
                    <p>Predicted Class:</p>
                    <p class="{label_class}">{pred_label}</p>
                    <p><b>Confidence:</b> {confidence * 100:.2f}%</p>
                    <p><b>Average frame score:</b> {avg_score:.4f}</p>
                    <p><b>Frames analysed:</b> {frames}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(confidence))

            st.markdown("### 🧩 4. Sampled Video Frames")

            cols = st.columns(3)
            for i, (frame, prob) in enumerate(sampled_frames):
                frame_label = "Incident" if prob >= 0.5 else "Non-Incident"

                with cols[i % 3]:
                    st.image(
                        frame,
                        caption=f"Frame {i+1}: {frame_label} | Score: {prob:.4f}",
                        use_container_width=True
                    )

            st.markdown("### 🧠 5. Video Grad-CAM Explanation")

            st.write(
                "Grad-CAM is shown for selected sampled frames from the uploaded video. "
                "Since this project uses a frame-based video classification method, "
                "the explanation is also frame-based."
            )

            gradcam_cols = st.columns(3)

            for i, (gradcam_frame, prob) in enumerate(sampled_gradcams):
                frame_label = "Incident" if prob >= 0.5 else "Non-Incident"

                with gradcam_cols[i % 3]:
                    if gradcam_frame is not None:
                        st.image(
                            gradcam_frame,
                            caption=f"Grad-CAM Frame {i+1}: {frame_label} | Score: {prob:.4f}",
                            use_container_width=True
                        )
                    else:
                        st.warning(f"Grad-CAM could not be generated for frame {i+1}.")

        if os.path.exists(temp_path):
            os.remove(temp_path)


# =========================
# About section
# =========================

st.markdown("---")

st.subheader("About This Prototype")

st.write(
    """
    This system was trained using publicly available Kaggle datasets mapped into two classes:
    **Incident** and **Non-Incident**.

    For videos, the system extracts selected frames, classifies each frame, and averages the predictions
    to produce a final video-level classification.

    The Grad-CAM heatmap provides a visual explanation of which image regions, or selected video-frame regions, influenced the prediction.
    """
)

st.markdown(
    """
    <div class="footer">
        MCS AI Project | Community Incident Media Classification | Academic prototype only<br>
        This prototype is for academic demonstration and decision-support discussion only.
    </div>
    """,
    unsafe_allow_html=True
)
