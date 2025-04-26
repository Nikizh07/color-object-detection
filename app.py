import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import io # Needed for processing uploaded file bytes

# Import functions and variables from object_detector.py
from object_detector import (
    load_yolo_model,
    analyze_image,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Object Detector UI", layout="wide")

st.title("📦 Object Color & Type Detector")
st.write("""
Use your webcam or upload an image file. Click **Analyze** to detect objects,
identify their dominant color, and see the summary.
Providing a new image (capture or upload) clears previous analysis results.
""")
st.markdown("---") # Separator

# --- Session State Initialization ---
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None # Holds the OpenCV image (BGR) ready for analysis
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'individual_detections' not in st.session_state:
    st.session_state.individual_detections = None
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'image_source_caption' not in st.session_state:
    st.session_state.image_source_caption = "No image provided yet."
# --- End Session State ---


# --- Load Model ---
try:
    model, class_names = load_yolo_model(MODEL_PATH)
    logging.info("Model loading process initiated via Streamlit app.")
except SystemExit:
    logging.error("SystemExit called due to model loading failure. Streamlit app will stop.")
    st.error("Application halted due to critical model loading error. Check logs.")
    st.stop()
except Exception as e:
    logging.error(f"Unexpected error during model loading in app.py: {e}", exc_info=True)
    st.error(f"An unexpected error occurred during model loading: {e}")
    st.stop()
# --- End Load Model ---


# --- Sidebar Controls ---
st.sidebar.header("⚙️ Input & Actions")

# --- Webcam Input ---
st.sidebar.subheader("Webcam Input")
device_index = st.sidebar.number_input("Camera Device Index", min_value=0, max_value=10, value=0, step=1,
                                       help="Enter the index of your webcam (often 0 or 1).")
capture_button = st.sidebar.button("📸 Capture Image", key="capture")
st.sidebar.markdown("---") # Separator

# --- File Upload Input ---
st.sidebar.subheader("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload an image",
                                        type=["png", "jpg", "jpeg", "bmp", "webp"],
                                        key="uploader",
                                        help="Upload an image file for analysis.")
st.sidebar.markdown("---") # Separator

# --- Analysis Action ---
st.sidebar.subheader("Analysis")
analyze_button = st.sidebar.button("🔍 Analyze Image", key="analyze", type="primary",
                                  # Disable button if no image is loaded in session state
                                  disabled=(st.session_state.captured_image is None),
                                  help="Detect objects in the provided image (captured or uploaded).")
st.sidebar.markdown("---") # Separator

# --- Model Info ---
st.sidebar.header("📊 Model Info")
st.sidebar.caption(f"Using Model: `{MODEL_PATH.split('/')[-1]}`")
st.sidebar.caption(f"Confidence Threshold: `{CONFIDENCE_THRESHOLD}`")
st.sidebar.markdown("---")
# --- End Sidebar Controls ---


# --- Input Handling Logic ---

def clear_analysis_results():
    """Clears previous analysis results from session state."""
    st.session_state.annotated_image = None
    st.session_state.individual_detections = None
    st.session_state.object_counts = None
    st.session_state.error_message = None # Also clear errors when new input is provided


# Webcam Capture Action
if capture_button:
    clear_analysis_results()
    st.session_state.captured_image = None # Clear any previous image (capture or upload)
    logging.info(f"Capture button clicked. Attempting capture from device {device_index}.")

    with st.spinner(f"Connecting to camera {device_index}..."):
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            err_msg = f"Error: Could not open camera device {device_index}. Is it connected and not in use?"
            st.session_state.error_message = err_msg
            logging.error(err_msg)
        else:
            ret, frame = cap.read()
            cap.release()
            logging.info(f"Camera {device_index} released.")

            if not ret or frame is None:
                err_msg = f"Error: Failed to grab frame from device {device_index}."
                st.session_state.error_message = err_msg
                logging.error(err_msg)
            else:
                st.session_state.captured_image = frame # Store BGR frame
                st.session_state.image_source_caption = f"Image captured from Webcam {device_index}"
                logging.info(f"Successfully captured frame from device {device_index}. Shape: {frame.shape}")
                st.toast("Image captured successfully!", icon="📸")
                # Rerun to update the analyze button state immediately
                st.rerun()


# File Upload Action (Handled implicitly when uploaded_file changes)
if uploaded_file is not None:
    # Check if this is a new upload compared to the last run potentially
    # This basic check might not be perfect but prevents reprocessing the same upload on unrelated reruns
    if 'last_uploaded_filename' not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
        clear_analysis_results()
        st.session_state.captured_image = None # Clear any previous image
        logging.info(f"File uploaded: {uploaded_file.name} (Type: {uploaded_file.type}, Size: {uploaded_file.size} bytes)")

        try:
            # Read file bytes
            bytes_data = uploaded_file.getvalue()
            # Convert bytes to numpy array
            file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
            # Decode image using OpenCV
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Read as BGR

            if opencv_image is None:
                err_msg = f"Error: Could not decode uploaded file '{uploaded_file.name}'. Is it a valid image format?"
                st.session_state.error_message = err_msg
                logging.error(err_msg)
                st.session_state.last_uploaded_filename = None # Reset tracker on error
            else:
                st.session_state.captured_image = opencv_image # Store BGR image
                st.session_state.image_source_caption = f"Image uploaded: {uploaded_file.name}"
                st.session_state.last_uploaded_filename = uploaded_file.name # Track successful upload
                logging.info(f"Successfully decoded uploaded image '{uploaded_file.name}'. Shape: {opencv_image.shape}")
                st.toast("Image uploaded successfully!", icon="📄")
                # Rerun to update the UI and analyze button state
                st.rerun()

        except Exception as e:
            err_msg = f"Error processing uploaded file '{uploaded_file.name}': {e}"
            st.session_state.error_message = err_msg
            st.session_state.last_uploaded_filename = None # Reset tracker on error
            logging.error(err_msg, exc_info=True)


# Analyze Button Action
if analyze_button:
    if st.session_state.captured_image is not None:
        # Clear only error message before starting analysis
        st.session_state.error_message = None
        logging.info("Analyze button clicked. Starting image analysis.")
        with st.spinner("🧠 Analyzing image... Please wait."):
            # Use the imported function name analyze_image
            annotated_img, ind_detect, obj_counts = analyze_image(
                st.session_state.captured_image, model, class_names
            )
            # Store results
            st.session_state.annotated_image = annotated_img
            st.session_state.individual_detections = ind_detect
            st.session_state.object_counts = obj_counts

            if annotated_img is None: # Should be rare now
                 err_msg = "Image analysis failed unexpectedly. Check logs."
                 st.session_state.error_message = err_msg
                 logging.error(err_msg)
            else:
                 logging.info("Image analysis finished.")
                 st.toast("Analysis complete!", icon="✨")
                 # Rerun to update the display immediately after analysis
                 st.rerun()

    else:
        # Should not happen due to button disable logic, but good to have
        st.warning("Cannot analyze - no image provided (capture or upload).")
        logging.warning("Analyze button clicked but no image was available in session state.")

# --- End Input Handling Logic ---


# --- Main Area Display ---

# Display Error Messages if any occurred
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# Display Images
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Image Preview") # Changed subheader
    if st.session_state.captured_image is not None:
        # Convert BGR (OpenCV default) to RGB for Streamlit display
        st.image(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB),
                 caption=st.session_state.image_source_caption, # Use dynamic caption
                 use_container_width=True)
    else:
        st.info("Provide an image via Webcam Capture or File Upload in the sidebar.")

with col2:
    st.subheader("🎨 Analysis Result")
    if st.session_state.annotated_image is not None:
         # Convert BGR (from OpenCV drawing) to RGB for Streamlit display
        st.image(cv2.cvtColor(st.session_state.annotated_image, cv2.COLOR_BGR2RGB),
                 caption="Detections from Last Analysis",
                 use_container_width=True)
    elif st.session_state.captured_image is not None:
         st.info("Image provided, but not analyzed yet. Click 'Analyze Image'.")
    else:
         st.info("Provide and analyze an image.")

st.markdown("---") # Separator

# Display Detection Details and Summary
col3, col4 = st.columns(2)

with col3:
    st.subheader("📋 Detected Objects List")
    if st.session_state.individual_detections is not None:
        if st.session_state.individual_detections:
            df_detections = pd.DataFrame(st.session_state.individual_detections)
            st.dataframe(df_detections, use_container_width=True, height=300)
        else:
            if st.session_state.annotated_image is not None:
                 st.info(f"No objects detected meeting the {CONFIDENCE_THRESHOLD} confidence threshold.")
            else:
                 st.info("Analysis not performed yet.")
    else:
        st.info("Analysis not performed yet.")

with col4:
    st.subheader("📊 Detection Summary")
    if st.session_state.object_counts is not None:
        if st.session_state.object_counts:
            summary_list = []
            sorted_counts = sorted(st.session_state.object_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
            for (color, name), count in sorted_counts:
                plural = "s" if count > 1 else ""
                summary_list.append(f"- **{count}** {color.capitalize()} {name.capitalize()}{plural}")
            st.markdown("\n".join(summary_list))
        else:
            if st.session_state.annotated_image is not None:
                 st.info(f"No objects detected meeting the {CONFIDENCE_THRESHOLD} confidence threshold.")
            else:
                 st.info("Analysis not performed yet.")

    else:
        st.info("Analysis not performed yet.")
# --- End Main Area Display ---
