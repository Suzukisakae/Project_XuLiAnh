# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run pages/main.py
import streamlit as st
from PIL import Image


import style


def style_transfer():
    st.title('PyTorch Style Transfer')
    
    col1, col2 = st.columns([1, 2])  # Bạn có thể điều chỉnh tỷ lệ cột tùy ý
    with col1:
        style_name = st.selectbox(
            'Select Style',
            ('candy', 'mosaic', 'rain_princess', 'udnie')
        )
    with col2:
        image_style_path = "./pages/style_transfer/images/style-images/" + style_name + ".jpg"
        image_style = Image.open(image_style_path)
        st.image(image_style, width=200)

    model= "./pages/style_transfer/saved_models/" + style_name + ".pth"
    input_image = st.file_uploader("Upload an image", type=["bmp", "png", "jpg", "jpeg"])
    output_image = "./Image/output-images/" + style_name + "-result.jpg"

    if input_image is not None:
        st.write('### Source image:')
        image = Image.open(input_image)
        st.image(image, width=400) # image: numpy array
        clicked = st.button('Stylize')

        if clicked:
            model = style.load_model(model)
            style.stylize(model, input_image, output_image)

            st.write('### Output image:')
            image = Image.open(output_image)
            st.image(image, width=400)
