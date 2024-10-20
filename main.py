from PIL import Image
import streamlit as st
import os
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)


def process_image(input_image, prompt, num_inference_steps, image_guidance_scale):
    model_id = "timbrooks/instruct-pix2pix"

    # initializing pipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, safety_checker=None
    )
    pipe.to("cpu")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    images = pipe(
        prompt,
        image=input_image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
    ).images

    # (optional) save image to local directory
    images[0].save(os.path.join("output/created_image.png"))

    return images[0]


def render_ui():
    # Streamlit UI
    st.set_page_config(page_title="Image Transformer", layout="centered")

    # Custom styles for the UI
    st.markdown(
        """
        <style>
            .header {
                text-align: center;
                color: #4F8A8B;
                font-family: 'Arial', sans-serif;
                font-size: 2.5em;
                margin-top: -50px;
            }
            .description {
                text-align: center;
                color: #666;
                font-family: 'Arial', sans-serif;
                font-size: 1.1em;
                margin-bottom: 30px;
            }
            .dropzone {
                height: 400px;  /* Increased height for the droppable area */
                border: 2px dashed #4F8A8B;
                border-radius: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
                color: #4F8A8B;
                font-size: 1.5em;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Title and Description
    st.markdown('<h1 class="header">Image Transformer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="description">Transform your images with customized prompts and settings!</p>',
        unsafe_allow_html=True,
    )

    # Section for Image Upload
    uploaded_image = st.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png"], label_visibility="visible"
    )

    # Placeholder for sections
    if uploaded_image:
        input_image = Image.open(uploaded_image).resize((720, 720))
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

        # Scroll to prompt section after image upload
        st.markdown('<div id="prompt-section"></div>', unsafe_allow_html=True)
        st.subheader("Adjust Your Settings")
        prompt = st.text_input("Enter a prompt for image transformation:")
        quality = st.slider(
            "Quality (Number of Inference Steps)", min_value=5, max_value=20, value=10
        )
        image_guidance_scale = st.slider(
            "Image Likeness", min_value=0.5, max_value=2.0, value=1.0
        )

        # Button to Generate Image
        generate_button = st.button("Generate Image")

        if generate_button:
            # Process the image
            output_image = process_image(
                input_image, prompt, quality, image_guidance_scale
            )

            # Scroll to output section after generating the image
            st.markdown('<div id="output-section"></div>', unsafe_allow_html=True)
            st.subheader("Transformed Image")
            st.image(output_image, caption="Processed Image", use_column_width=True)


if __name__ == "__main__":
    render_ui()
