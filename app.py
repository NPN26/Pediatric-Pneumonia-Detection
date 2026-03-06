"""Streamlit app for pediatric pneumonia prediction from chest X-rays."""

from __future__ import annotations

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from model_def import load_model, predict_with_gradcam

st.set_page_config(
    page_title="Pediatric Pneumonia Detector",
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
    st.title("Pediatric Pneumonia Detection")
    st.caption("Upload one chest X-ray image (`jpg`, `jpeg`, or `png`) to run inference.")

    with st.expander("Important disclaimer", expanded=True):
        st.warning(DISCLAIMER)

    uploaded_file = st.file_uploader(
        "Upload chest X-ray image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload an image file to see prediction and Grad-CAM.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a JPG or PNG chest X-ray.")
        return
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not read the file: {exc}")
        return

    model = get_model()
    model.eval()

    pred = predict_with_gradcam(model=model, image=image, device="cpu")
    overlay = pred["gradcam_overlay"]

    st.subheader("Prediction")
    confidence_pct = pred["confidence"] * 100
    if pred["label"] == "PNEUMONIA":
        st.error(f"Predicted class: **{pred['label']}** ({confidence_pct:.2f}% confidence)")
    else:
        st.success(f"Predicted class: **{pred['label']}** ({confidence_pct:.2f}% confidence)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(image, caption="Uploaded X-ray", width='stretch')
    with col_b:
        st.image(overlay, caption="Grad-CAM heatmap overlay", width='stretch')

    st.subheader("Class probabilities")
    st.write({label: f"{value * 100:.2f}%" for label, value in pred["probabilities"].items()})


if __name__ == "__main__":
    main()
