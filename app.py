# app.py (Streamlit UI using Device Index)
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

from object_detector import (
    load_yolo_model,
    # capture_frame_from_stream is removed
    analyze_image,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD
)


st.set_page_config(page_title="Object Detector UI", layout="wide")

st.title("📦 Object Color & Type Detector (Webcam)")
st.write("""
Select your camera device index (usually 0 or 1). Capture an image.
Then, click Analyze to detect objects, identify their dominant color, and see the summary.
Capturing a new image will clear previous analysis results.
""")


if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'individual_detections' not in st.session_state:
    st.session_state.individual_detections = None
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None


try:
    model, class_names = load_yolo_model(MODEL_PATH)
except Exception as e:
    st.error(f"Fatal Error: Could not load YOLO model. Error: {e}")
    st.stop()


st.sidebar.header("Input & Actions")
# Input for device index instead of URL
device_index = st.sidebar.number_input("Camera Device Index", min_value=0, max_value=10, value=0, step=1)

capture_button = st.sidebar.button("Capture Image", key="capture")
analyze_button = st.sidebar.button("Analyze Captured Image", key="analyze", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(f"Using Model: {MODEL_PATH.split('/')[-1]}")
st.sidebar.caption(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")


# Capture Button Logic (Modified for device index)
if capture_button:
    st.session_state.error_message = None # Clear previous errors
    st.session_state.captured_image = None # Clear previous image
    st.session_state.annotated_image = None # Clear previous analysis
    st.session_state.individual_detections = None
    st.session_state.object_counts = None

    with st.spinner(f"Attempting to capture from device {device_index}..."):
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            st.session_state.error_message = f"Error: Could not open camera device {device_index}. Is it connected/available?"
        else:
            # Allow camera to warm up slightly (optional)
            # time.sleep(0.5) # Requires 'import time' at the top
            ret, frame = cap.read()
            cap.release() # Release camera immediately after capture

            if not ret or frame is None:
                st.session_state.error_message = f"Error: Failed to grab frame from device {device_index}."
            else:
                st.session_state.captured_image = frame # Store the captured frame
                print(f"Successfully captured frame from device {device_index}") # For debugging


# Analyze Button Logic (Remains the same, depends on session state)
if analyze_button:
    if st.session_state.captured_image is not None:
        st.session_state.error_message = None # Clear previous errors
        with st.spinner("Analyzing image..."):
            annotated_img, ind_detect, obj_counts = analyze_image(
                st.session_state.captured_image, model, class_names
            )
            st.session_state.annotated_image = annotated_img
            st.session_state.individual_detections = ind_detect
            st.session_state.object_counts = obj_counts
            if annotated_img is None: # If analysis itself failed
                 st.session_state.error_message = "Image analysis failed."
    else:
        st.warning("Please capture an image first before analyzing.")
        st.session_state.error_message = "Analysis attempted without a captured image."


# --- Main Area for Display (Remains largely the same) ---

if st.session_state.error_message:
    st.error(st.session_state.error_message)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Captured Image")
    if st.session_state.captured_image is not None:
        st.image(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB),
                 caption="Most Recent Capture",
                 use_container_width=True)
    else:
        st.info("No image captured yet.")

with col2:
    st.subheader("Analysis Result (Boxes Only)")
    if st.session_state.annotated_image is not None:
        st.image(cv2.cvtColor(st.session_state.annotated_image, cv2.COLOR_BGR2RGB),
                 caption="Detections from last analysis",
                 use_container_width=True)
    else:
        st.info("Image not analyzed yet, or analysis cleared.")


st.markdown("---")


st.subheader("Detected Objects List")
if st.session_state.individual_detections is not None:
    if st.session_state.individual_detections:
        df_detections = pd.DataFrame(st.session_state.individual_detections)
        st.dataframe(df_detections, use_container_width=True)
    else:
        st.info("No objects detected in the last analysis.")
else:
    st.info("Analysis not performed yet.")

st.subheader("Detection Summary")
if st.session_state.object_counts is not None:
    if st.session_state.object_counts:
        summary_list = []
        sorted_counts = sorted(st.session_state.object_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        for (color, name), count in sorted_counts:
            plural = "s" if count > 1 else ""
       
            summary_list.append(f"- {count} {color.capitalize()} {name.capitalize()}{plural}")
        st.markdown("\n".join(summary_list))
    else:
        st.info("No objects detected in the last analysis.")
else:
    st.info("Analysis not performed yet.")
