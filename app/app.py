"""App"""
import streamlit as st
from PIL import Image

from executor.inferrer import Inferrer

configfile = 'configs/default.yml'
inferrer = Inferrer(configfile)

st.title("Image Classification App")
st.write('\n')

image = Image.open('./resources/image.png')
show = st.image(image, use_column_width=True)

st.sidebar.title('Upload Image')

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.sidebar.file_uloader(' ', type=['png', 'jpg', 'jpeg'])
u_img = None

if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)

st.sidebar.write('/n')

if st.sidebar.button('Click Here to Classify'):
    if uploaded_file is None:
        st.sidebar.write('Please upload an Image to Classify')
    else:
        with st.spinner('Classifying..'):
            results = inferrer.infer(u_img)
            st.success('Done!')

        st.sidebar.header('Algorithm Predicts: ')
        probability = '{:.3f}'.format(results['prob'] * 100)

        class_name = inferrer.classes[results['label']]
        st.sidebar.write(f"'{class_name.capitalize()}' picture.\n")
        st.sidebar.write(f"Probability: {probability}%")
