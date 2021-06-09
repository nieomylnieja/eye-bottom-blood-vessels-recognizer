from os import listdir
from typing import Optional

import streamlit as st

from classifiers import Dataset, ImageType
from retinal_image import RetinalImage
from config import Config
from recognizers import FilterRecognizer

hsv_v_th_help = "This helps mitigate light spots created when taking the photos"
vis_cor_help = "The resulting image will have the vessels and potentially also some noise more exposed"


def run_app():
    run_recognizer = st.button("Run blood vessels recognition")

    img: Optional[RetinalImage] = None

    hsv_v_threshold_input, visibility_correction_input = st.beta_columns(2)
    hsv_v_threshold = hsv_v_threshold_input.number_input("HSV V channel threshold",
                                                         value=0.3,
                                                         step=0.01,
                                                         help=hsv_v_th_help)
    visibility_correction = visibility_correction_input.number_input("Visibility enhancement multiplier",
                                                                     value=100000,
                                                                     step=1000,
                                                                     help=vis_cor_help)

    picker_container, img_picker_preview = st.beta_columns(2)
    picker_container.beta_container()
    dataset_name = picker_container.radio("Choose dataset type:", Dataset.list())
    if dataset_name:
        dataset = Dataset(dataset_name)
        images_path = Config.DatasetBasePath.joinpath(dataset).joinpath(ImageType.INPUT)
        images = sorted(listdir(images_path))
        img_name = picker_container.selectbox("Choose an image", images)
        if img_name:
            img = RetinalImage(dataset, img_name)
            img_picker_preview.image(img.open())

    if run_recognizer:
        with st.spinner("Running recognition..."):
            result = FilterRecognizer(img).run(hsv_v_threshold, visibility_correction)
            st.pyplot(result.plt)
            st.write("The table represents statistical summary of the recognition using four hit/miss tolerance thresholds")
            st.markdown("Thresholds are applied to the resulting image as: `if PIXEL > THRESHOLD then 1 else 0`")
            st.write(result.statistics)
        st.success("Done!")


if __name__ == "__main__":
    run_app()
