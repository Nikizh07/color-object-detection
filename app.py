import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

from object_detector import (
    load_yolo_model,
    capture_frame_from_stream,
    analyze_image,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD
)


st.set_page_config(page_title="Object Detector UI", layout="wide")

st.title(" Object Color & Type Detector")
st.write("Analyzes an image from an IP Camera to detect objects, identify their dominant color, and provide counts.")


try:
    model, class_names = load_yolo_model(MODEL_PATH)
except Exception as e:
    st.error(f"Fatal Error: Could not load YOLO model. Please check the model path and dependencies. Error: {e}")
    st.stop()


st.sidebar.header("Input Source")
default_ip_url = "http://10.40.40.203:4747/video"
ip_url = st.sidebar.text_input("IP Camera URL", value=default_ip_url)

analyze_button = st.sidebar.button("Capture & Analyze", type="primary")
st.sidebar.markdown("---")


if analyze_button:
    if not ip_url:
        st.warning("Please enter an IP Camera URL.")
    else:
        with st.spinner("Connecting to camera and capturing image..."):
            captured_image, error_msg = capture_frame_from_stream(ip_url)

        if error_msg:
            st.error(error_msg)
        elif captured_image is not None:
            st.success("Image captured successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Captured Image")
                st.image(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB), caption="Captured Frame", use_column_width=True)

            with st.spinner("Analyzing image... (This may take a moment)"):
                annotated_image, individual_detections, object_counts = analyze_image(
                    captured_image, model, class_names
                )

            if annotated_image is None:
                st.error("Image analysis failed.")
            else:
                with col2:
                    st.subheader("Annotated Image (Boxes Only)")
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detections", use_column_width=True)

                st.markdown("---")

                st.subheader("Detected Objects")
                if individual_detections:
                    df_detections = pd.DataFrame(individual_detections)
                    st.dataframe(df_detections, use_container_width=True)
                else:
                    st.info("No objects detected meeting the confidence threshold.")

                st.subheader("Detection Summary")
                if object_counts:
                    summary_list = []
                    sorted_counts = sorted(object_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
                    for (color, name), count in sorted_counts:
                        plural = "s" if count > 1 else ""
                        summary_list.append(f"- {count} {color.capitalize()} {name.capitalize()}{plural}")
                    st.markdown("\n".join(summary_list))
                else:
                     st.info("No objects detected meeting the confidence threshold.")
        else:
            st.error("Failed to capture image for unknown reasons.")

else:
    st.info("Enter the IP Camera URL in the sidebar and click 'Capture & Analyze'.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Using Model: {MODEL_PATH.split('/')[-1]}")
st.sidebar.caption(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")