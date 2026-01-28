import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Face Detection", layout="wide")
st.title("Face Detection Using YOLO")

# Load face detection model
model_path = hf_hub_download(
    repo_id="navkaur/Face_detection_using_YOLO",
    filename="yolov8n-face-lindevs.pt",
    repo_type="space"
)
model = YOLO(model_path)

st.success("✅ Model loaded successfully")

# Sidebar options
mode = st.sidebar.radio(
    "Select Mode",
    ["Image Upload", "Webcam (Local Only)"]
)

# ---------------- IMAGE UPLOAD ----------------
if mode == "Image Upload":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)

        results = model(img)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(img, caption="Detected Faces", use_container_width=True)

# ---------------- WEBCAM ----------------
elif mode == "Webcam (Local Only)":
    st.warning("⚠️ Webcam works only when running locally, not on Hugging Face.")

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

st.markdown("---")
st.write("👩‍💻 Developed by *Navjot Kaur*")
