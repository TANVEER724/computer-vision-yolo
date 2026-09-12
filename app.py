import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="AI Vision Analytics",
    page_icon="👁️",
    layout="wide"
)


# ==========================================
# Title
# ==========================================

st.title("👁️ AI Vision Analytics")

st.markdown(
    """
    Real-time computer vision system for
    object detection, tracking and analytics.
    """
)


# ==========================================
# Load model
# ==========================================

@st.cache_resource
def load_model():

    return YOLO("yolo11n.pt")


model = load_model()


# ==========================================
# Sidebar
# ==========================================

st.sidebar.header("⚙️ Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)


# ==========================================
# Input selection
# ==========================================

option = st.sidebar.selectbox(
    "Select Input",
    [
        "Image",
        "Video"
    ]
)


# ==========================================
# Image detection
# ==========================================

if option == "Image":

    st.subheader("📷 Image Object Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        image_array = np.array(image)


        # YOLO detection

        results = model(
            image_array,
            conf=confidence,
            verbose=False
        )


        # Draw detections

        output = results[0].plot()


        # Display

        col1, col2 = st.columns(2)


        with col1:

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )


        with col2:

            st.image(
                output,
                caption="Detected Objects",
                channels="BGR",
                use_container_width=True
            )


        # Detection statistics

        st.subheader("📊 Detection Results")

        result = results[0]


        if result.boxes is not None:

            count = len(
                result.boxes
            )

            st.metric(
                "Objects Detected",
                count
            )

else:

    st.subheader("🎥 Video Object Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov"
        ]
    )


    if uploaded_video is not None:

        video_bytes = uploaded_video.read()

        temp_path = "temp_video.mp4"

        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(video_bytes)


        cap = cv2.VideoCapture(
            temp_path
        )


        frame_placeholder = st.empty()


        while cap.isOpened():

            ret, frame = cap.read()


            if not ret:
                break


            results = model(
                frame,
                conf=confidence,
                verbose=False
            )


            output = results[0].plot()


            frame_placeholder.image(
                output,
                channels="BGR",
                use_container_width=True
            )


        cap.release()