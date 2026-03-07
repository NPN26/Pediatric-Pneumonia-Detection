"""Streamlit app for pediatric pneumonia prediction from chest X-rays."""

from __future__ import annotations

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from model_def import load_model, predict_with_gradcam

st.set_page_config(
    page_title="Pediatric Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
)

DISCLAIMER = (
    "For research/demo use only. This tool is not a medical device and must not be "
    "used as a standalone clinical diagnosis."
)


@st.cache_resource
def get_model() -> torch.nn.Module:
    return load_model(weights_path="best_model.pth", device="cpu")


def main() -> None:
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if 'last_uploaded_file' not in st.session_state:
        st.session_state['last_uploaded_file'] = None
    # status: "idle" | "running" | "done"
    if 'status' not in st.session_state:
        st.session_state['status'] = "idle"

    st.title("🫁 Pediatric Pneumonia Detection")
    st.caption("Upload one chest X-ray image (jpg, jpeg, or png) to run inference.")
    st.warning(f"⚠️ **Disclaimer** — {DISCLAIMER}")

    uploaded_file = st.file_uploader(
        "Upload chest X-ray image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload an image file to see prediction and Grad-CAM.")
        st.session_state.prediction = None
        st.session_state.status = "idle"
        return

    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if file_id != st.session_state['last_uploaded_file']:
        st.session_state['last_uploaded_file'] = file_id
        st.session_state.prediction = None
        st.session_state.status = "idle"

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("Not a valid image — please upload a JPG or PNG chest X-ray.")
        return
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return

    st.markdown("---")

    if st.session_state.status == "done":
        pred = st.session_state.prediction
        confidence_pct = pred["confidence"] * 100
        if pred["label"] == "PNEUMONIA":
            st.error(f"Predicted class: **{pred['label']}** ({confidence_pct:.2f}% confidence)")
        else:
            st.success(f"Predicted class: **{pred['label']}** ({confidence_pct:.2f}% confidence)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(image, caption="Uploaded X-ray", width='stretch')

    with col_b:
        if st.session_state.status == "idle":
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                if st.button("Run Prediction", key="predict_btn"):
                    st.session_state.status = "running"
                    st.rerun()

        elif st.session_state.status == "running":
            with st.spinner("Running inference…"):
                st.session_state.prediction = predict_with_gradcam(
                    model=get_model(), image=image, device="cpu"
                )
            st.session_state.status = "done"
            st.rerun()

        elif st.session_state.status == "done":
            st.image(st.session_state.prediction["gradcam_overlay"], caption="Grad-CAM heatmap overlay", width='stretch')

    if st.session_state.status == "done":
        st.subheader("Class probabilities")
        st.write({label: f"{value * 100:.2f}%" for label, value in st.session_state.prediction["probabilities"].items()})

if __name__ == "__main__":
    main()
