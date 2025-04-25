import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import time # Keep for potential future use

# --- Import functions and constants from your object_detector.py ---
try:
    from object_detector import (
        MODEL_PATH,
        CONFIDENCE_THRESHOLD, # Import the threshold from your file
        load_yolo_model,       # Import the function with caching included
        capture_frame_from_stream, # Import your capture function
        analyze_image          # Import your analysis function
        # No need to import get_dominant_color or classify_color directly,
        # as they are used internally by analyze_image
    )
    print("Successfully imported from object_detector.py")
except ImportError as e:
    st.error(f"Fatal Error: Could not import required elements from 'object_detector.py'. "
             f"Ensure the file exists in the same directory and all dependencies "
             f"(ultralytics, sklearn, etc.) are installed. Error details: {e}")
    st.stop()
except Exception as e:
     st.error(f"Fatal Error: An unexpected error occurred during initial import from object_detector.py. Error: {e}")
     st.stop()


# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'individual_detections' not in st.session_state:
    st.session_state.individual_detections = None
if 'object_counts' not in st.session_state:
    st.session_state.object_counts = None
if 'error_msg' not in st.session_state:
    st.session_state.error_msg = None
if 'info_msg' not in st.session_state:
    # Set the initial informational message
    st.session_state.info_msg = "Enter the IP Camera URL/Index in the sidebar and click 'Capture & Analyze'."

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Object Detector UI", layout="wide")

st.title("Object Color & Type Detector")
st.write("Analyzes an image from an IP Camera or device to detect objects, identify their dominant color, and provide counts.")

# --- Model Loading ---
# Use the cached function directly from object_detector.py
# The @st.cache_resource within load_yolo_model handles the caching.
try:
    # This will be cached by the decorator in object_detector.py
    yolo_model, class_names_map = load_yolo_model(MODEL_PATH)
    # Display success message or status in the app if needed
    st.sidebar.success(f"Model loaded: {MODEL_PATH.split('/')[-1]}")
except Exception as e:
    # Display error in the main app area if loading fails
    st.error(f"Fatal Error during Model Loading: {e}")
    st.error("The application cannot proceed without a valid model. Please check the model path and file integrity.")
    st.stop()


# --- Sidebar ---
st.sidebar.header("Input Source")
default_ip_url = 0 # Common default, adjust if your primary camera is different
ip_url = st.sidebar.text_input("IP Camera URL or Device Index", value=str(default_ip_url))

analyze_button = st.sidebar.button("Capture & Analyze", type="primary")
clear_button = st.sidebar.button("Clear Results")
st.sidebar.markdown("---")
st.sidebar.caption(f"Using Model: {MODEL_PATH.split('/')[-1]}") # Display model from constant
st.sidebar.caption(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}") # Display threshold from constant
# --- End Sidebar ---


# --- Button Logic ---

# Handle Clear Button Click
if clear_button:
    # Reset all state variables related to results and messages
    st.session_state.captured_image = None
    st.session_state.annotated_image = None
    st.session_state.individual_detections = None
    st.session_state.object_counts = None
    st.session_state.error_msg = None
    st.session_state.info_msg = "Results cleared. Enter URL/Index and click 'Capture & Analyze'."
    st.rerun()

# Handle Analyze Button Click
if analyze_button:
    # Clear previous state before starting new analysis
    st.session_state.captured_image = None
    st.session_state.annotated_image = None
    st.session_state.individual_detections = None
    st.session_state.object_counts = None
    st.session_state.error_msg = None
    st.session_state.info_msg = None

    ip_url_stripped = ip_url.strip()
    if not ip_url_stripped:
         st.session_state.error_msg = "Error: Please enter an IP Camera URL or device index."
    else:
        # Determine if it's likely an index or URL
        try:
            video_source = int(ip_url_stripped) # Try converting to integer for index
        except ValueError:
            video_source = ip_url_stripped # Keep as string if not an integer (URL)

        # Proceed with capture if input is provided
        with st.spinner("Connecting to camera and capturing image..."):
            # Use the imported capture function
            captured_image_bgr, error_msg_capture = capture_frame_from_stream(video_source)

        if error_msg_capture:
            st.session_state.error_msg = error_msg_capture # Store specific capture error
        elif captured_image_bgr is not None:
            st.session_state.captured_image = captured_image_bgr # Store BGR image
            st.session_state.info_msg = "Image captured successfully! Analyzing..."

            # Perform analysis only if capture was successful
            with st.spinner("Analyzing image..."):
                try:
                    # Call the imported analysis function
                    annotated_img, detections, counts = analyze_image(
                        captured_image_bgr, yolo_model, class_names_map # Pass the map
                    )
                    # Store results
                    st.session_state.annotated_image = annotated_img
                    st.session_state.individual_detections = detections
                    st.session_state.object_counts = counts
                    # Update info message based on results
                    if not detections:
                        st.session_state.info_msg = "Analysis complete. No objects detected meeting the confidence threshold."
                    else:
                        st.session_state.info_msg = f"Analysis complete! Found {len(detections)} object(s)."

                except Exception as analysis_error:
                     st.session_state.error_msg = f"Error during image analysis: {analysis_error}"
                     # Clear analysis-specific results on error
                     st.session_state.annotated_image = None
                     st.session_state.individual_detections = None
                     st.session_state.object_counts = None
                     # Keep the original captured image visible
                     print(f"Analysis Exception: {analysis_error}") # Log detailed error to console


        else:
            # Fallback if capture returns None without a message (shouldn't happen with current object_detector.py)
            st.session_state.error_msg = f"Failed to capture image from '{video_source}' (received null frame)."

    # Rerun to display the updated state
    st.rerun()


# --- Main Page Display Area ---

# Display Status Messages (Error or Info)
if st.session_state.error_msg:
    st.error(st.session_state.error_msg)
elif st.session_state.info_msg:
    st.info(st.session_state.info_msg)

# Display Images (converting BGR to RGB for st.image)
if st.session_state.captured_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Captured Image")
        try:
            st.image(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB), caption="Captured Frame", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying original image: {e}")

    with col2:
        if st.session_state.annotated_image is not None:
            st.subheader("Annotated Image")
            try:
                 # Annotated image from analyze_image should also be BGR
                 st.image(cv2.cvtColor(st.session_state.annotated_image, cv2.COLOR_BGR2RGB), caption="Detections", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying annotated image: {e}")
                st.write("Annotated image data shape:", st.session_state.annotated_image.shape if st.session_state.annotated_image is not None else 'None')

        elif st.session_state.error_msg and "analysis" in st.session_state.error_msg.lower():
             st.subheader("Annotated Image")
             st.warning("Analysis could not be performed due to an error.")
        elif st.session_state.captured_image is not None: # If capture ok, but no annotation yet or no detections
            st.subheader("Annotated Image")
            if st.session_state.individual_detections is None and not st.session_state.error_msg:
                # This case might occur briefly during processing or if analysis hasn't run
                 st.info("Awaiting analysis or no results to display.")
            # If detections is an empty list after analysis (no objects found), this space remains blank unless handled otherwise


# Display Detection Details (DataFrame)
if st.session_state.individual_detections is not None:
    st.markdown("---")
    st.subheader("Detected Objects")
    if st.session_state.individual_detections: # Check if the list has items
        try:
            # Check the keys in the first dictionary to ensure they match DataFrame columns
            # Expected keys from analyze_image: "Color", "Object", "Confidence"
            df_detections = pd.DataFrame(st.session_state.individual_detections)
            # Optional: Reorder columns if needed
            # df_detections = df_detections[["Object", "Color", "Confidence"]]
            st.dataframe(df_detections, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error displaying detection details table: {e}")
            st.write("Raw detection data:", st.session_state.individual_detections)
    # Display message if analysis ran but found nothing
    elif not st.session_state.error_msg: # Check analysis completed without error
        st.info("No objects detected meeting the confidence threshold.")

# Display Detection Summary
if st.session_state.object_counts is not None:
    # Display header only if there are counts or if no detections message should appear here
     if st.session_state.object_counts or (st.session_state.individual_detections is not None and not st.session_state.individual_detections):
          st.subheader("Detection Summary")

     if st.session_state.object_counts: # Check if the dictionary has items
        summary_list = []
        try:
            # Keys are (color_name, class_name)
            valid_counts = {k: v for k, v in st.session_state.object_counts.items() if isinstance(k, tuple) and len(k) == 2}
            # Sort by count desc, then color, then name
            sorted_counts = sorted(valid_counts.items(), key=lambda item: (-item[1], str(item[0][0]), str(item[0][1])))

            for (color, name), count in sorted_counts:
                plural = "s" if count > 1 else ""
                # Use capitalize directly as analyze_image might not do it
                summary_list.append(f"- {count} {str(color).capitalize()} {str(name).capitalize()}{plural}")

            if summary_list:
                st.markdown("\n".join(summary_list))

        except Exception as e:
            st.warning(f"Could not generate summary list. Error: {e}")
            st.write("Raw counts data:", st.session_state.object_counts)

    # Display message if analysis ran but found nothing (consistent with table message)
     elif st.session_state.individual_detections is not None and not st.session_state.individual_detections and not st.session_state.error_msg:
         st.info("No objects detected meeting the confidence threshold.")

# Footer or final checks can go here if needed