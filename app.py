import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import time

# --- Import functions and constants from your object_detector.py ---
try:
    from object_detector import (
        MODEL_PATH,
        CONFIDENCE_THRESHOLD,
        load_yolo_model,
        analyze_image
        # We will implement a simplified capture here
    )
    print("Successfully imported core elements from object_detector.py")
except ImportError as e:
    st.error(f"Fatal Error: Could not import required elements from 'object_detector.py'. "
             f"Ensure the file exists and dependencies are installed. Error details: {e}")
    st.stop()
except Exception as e:
     st.error(f"Fatal Error: An unexpected error occurred during initial import. Error: {e}")
     st.stop()

# --- Simplified Frame Capture Function (not using object_detector's) ---
def capture_single_frame(source):
    """Opens camera, captures one frame, and releases immediately."""
    error_msg = None
    frame = None
    cap = None # Initialize cap to None
    try:
        try:
            source_int = int(source)
            print(f"Attempting to open camera index: {source_int}")
            cap = cv2.VideoCapture(source_int, cv2.CAP_DSHOW) # Try DSHOW for Windows
        except ValueError:
            source_url = str(source)
            print(f"Attempting to open camera URL: {source_url}")
            cap = cv2.VideoCapture(source_url)

        # Increased wait time slightly, might help slower cameras
        time.sleep(1.5) # Allow time for camera initialization

        if cap is None or not cap.isOpened():
            # Try common alternatives if initial attempt failed (for indices)
            if 'source_int' in locals():
                 print(f"Warning: Could not open camera index {source_int}. Trying fallbacks...")
                 for idx in [0, 1, 2, 3]:
                      if idx == source_int: continue
                      if cap: cap.release() # Release previous attempt
                      print(f"Trying index {idx}...")
                      cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                      time.sleep(0.5) # Shorter wait for fallbacks
                      if cap.isOpened():
                           print(f"Success: Opened camera index {idx} as fallback.")
                           source = idx # Update source to the working index
                           break
            # Check again after potential fallbacks
            if cap is None or not cap.isOpened():
                 error_msg = f"Error: Could not open video source '{source}'. Checked common indices/URL."
                 print(error_msg)
                 return None, error_msg

        # If opened successfully
        print(f"Camera source '{source}' opened successfully. Reading frame...")
        ret, frame = cap.read()

        if not ret or frame is None:
            error_msg = f"Error: Could not read frame from video source '{source}'. Camera might be busy or frame empty."
            print(error_msg)
            # Ensure release even if read fails
            if cap is not None and cap.isOpened():
                cap.release()
            return None, error_msg
        else:
             print("Frame captured successfully.")
             return frame.copy(), None # Return a copy of the frame and no error

    except Exception as e:
        error_msg = f"Error during frame capture process: {e}"
        print(error_msg)
        return None, error_msg
    finally:
        # Ensure camera is always released if it was opened
        if cap is not None and cap.isOpened():
            cap.release()
            print(f"Camera source '{source}' released.")


# --- Session State Initialization ---
if 'captured_frame_for_analysis' not in st.session_state:
    st.session_state.captured_frame_for_analysis = None
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'individual_detections' not in st.session_state:
    st.session_state.individual_detections = None
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = None
if 'error_msg' not in st.session_state:
    st.session_state.error_msg = None
if 'info_msg' not in st.session_state:
    st.session_state.info_msg = "Select camera source and click 'Capture Frame'."

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Object Detector", layout="wide")

st.title("Object Color & Type Detector")
st.write("Capture a frame from your camera, then analyze it.")

# --- Model Loading ---
try:
    # load_yolo_model already has @st.cache_resource in object_detector.py
    yolo_model, class_names_map = load_yolo_model(MODEL_PATH)
    model_loaded = yolo_model is not None
except Exception as e:
    st.error(f"Fatal Error during Model Loading: {e}")
    st.error("Cannot proceed without a model.")
    model_loaded = False
    st.stop() # Stop execution if model loading fails

if model_loaded:
    st.sidebar.success(f"Model loaded: {MODEL_PATH.split('/')[-1]}")

# --- Sidebar ---
st.sidebar.header("Camera Settings")
default_ip_url = 2
ip_url = st.sidebar.text_input("Camera URL or Device Index", value=str(default_ip_url))

st.sidebar.markdown("---")
st.sidebar.caption(f"Using Model: {MODEL_PATH.split('/')[-1]}")
st.sidebar.caption(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
st.sidebar.markdown("---")

# --- Main Area Layout ---
col_capture, col_analyze = st.columns(2)
with col_capture:
    capture_button = st.button("📷 Capture Frame", type="secondary", key="capture")
with col_analyze:
    analyze_button = st.button("✨ Analyze Captured Frame", type="primary",
                               disabled=st.session_state.captured_frame_for_analysis is None, key="analyze")

results_placeholder = st.container()

# --- Button Logic ---

if capture_button:
    st.session_state.error_msg = None
    st.session_state.info_msg = f"Attempting to capture frame from source: {ip_url.strip()}..."
    st.session_state.captured_frame_for_analysis = None
    st.session_state.annotated_image = None
    st.session_state.individual_detections = None
    st.session_state.object_counts = None

    source = ip_url.strip()
    if not source:
        st.session_state.error_msg = "Error: Please enter a camera source URL or Index."
        st.session_state.info_msg = None
    else:
        with st.spinner(f"Accessing camera '{source}'..."):
            frame, error = capture_single_frame(source)

        if error:
            st.session_state.error_msg = error
            st.session_state.info_msg = None
        elif frame is not None:
            st.session_state.captured_frame_for_analysis = frame
            st.session_state.info_msg = "Frame captured. Click 'Analyze Captured Frame'."
            print("Frame stored in session state.")
        else:
            st.session_state.error_msg = f"Capture failed for '{source}' (Unknown reason, null frame)."
            st.session_state.info_msg = None
    st.rerun()

if analyze_button:
    st.session_state.error_msg = None
    if st.session_state.captured_frame_for_analysis is not None:
        st.session_state.info_msg = "Analyzing captured frame..."
        st.session_state.annotated_image = None
        st.session_state.individual_detections = None
        st.session_state.object_counts = None

        with st.spinner("Analyzing..."):
            try:
                annotated_img, detections, counts = analyze_image(
                    st.session_state.captured_frame_for_analysis, yolo_model, class_names_map
                )
                st.session_state.annotated_image = annotated_img
                st.session_state.individual_detections = detections
                st.session_state.object_counts = counts
                if not detections:
                     st.session_state.info_msg = "Analysis complete. No objects detected meeting the threshold."
                else:
                     st.session_state.info_msg = f"Analysis complete! Found {len(detections)} object(s)."
                print("Analysis complete.")
            except Exception as analysis_error:
                 st.session_state.error_msg = f"Error during image analysis: {analysis_error}"
                 st.session_state.info_msg = None
                 st.session_state.annotated_image = None
                 st.session_state.individual_detections = None
                 st.session_state.object_counts = None
                 print(f"Analysis Exception: {analysis_error}")
    else:
        st.session_state.error_msg = "No frame available. Capture a frame first."
        st.session_state.info_msg = None
    st.rerun()

# --- Display Status and Results Area ---
with results_placeholder:
    st.markdown("---")
    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)
    elif st.session_state.info_msg:
        st.info(st.session_state.info_msg)

    if st.session_state.captured_frame_for_analysis is not None:
        if st.session_state.annotated_image is not None:
            st.subheader("📷 Captured Frame & ✨ Analysis Result")
            res_col1, res_col2 = st.columns(2)
            try:
                with res_col1:
                    st.caption("Original Captured")
                    # Use use_container_width instead of use_column_width
                    st.image(cv2.cvtColor(st.session_state.captured_frame_for_analysis, cv2.COLOR_BGR2RGB), use_container_width=True)
                with res_col2:
                    st.caption("Annotated Result")
                    # Use use_container_width instead of use_column_width
                    st.image(cv2.cvtColor(st.session_state.annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            except Exception as e:
                 st.error(f"Error displaying images: {e}")
        else:
            st.subheader("📷 Captured Frame Preview")
            try:
                # Use use_container_width instead of use_column_width
                st.image(cv2.cvtColor(st.session_state.captured_frame_for_analysis, cv2.COLOR_BGR2RGB), use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying captured frame: {e}")

        if st.session_state.individual_detections is not None or st.session_state.object_counts is not None:
            st.markdown("---")
            st.subheader("📊 Analysis Details")

            if st.session_state.individual_detections:
                try:
                    df_detections = pd.DataFrame(st.session_state.individual_detections)
                    # Use use_container_width instead of use_column_width
                    st.dataframe(df_detections, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Error displaying detection details table: {e}")
                    st.write("Raw detection data:", st.session_state.individual_detections)
            elif not st.session_state.error_msg and st.session_state.individual_detections is not None:
                st.info("No objects detected meeting the confidence threshold.")

            if st.session_state.object_counts:
                summary_list = []
                try:
                    valid_counts = {k: v for k, v in st.session_state.object_counts.items() if isinstance(k, tuple) and len(k) == 2}
                    sorted_counts = sorted(valid_counts.items(), key=lambda item: (-item[1], str(item[0][0]), str(item[0][1])))
                    if sorted_counts:
                        st.subheader("📈 Summary")
                        for (color, name), count in sorted_counts:
                            plural = "s" if count > 1 else ""
                            summary_list.append(f"- {count} {str(color).capitalize()} {str(name).capitalize()}{plural}")
                        if summary_list:
                            st.markdown("\n".join(summary_list))
                except Exception as e:
                    st.warning(f"Could not generate summary list. Error: {e}")
                    st.write("Raw counts data:", st.session_state.object_counts)
            elif not st.session_state.error_msg and st.session_state.object_counts is not None:
                pass # Message handled by individual_detections check