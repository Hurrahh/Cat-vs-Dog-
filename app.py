from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np
from PIL import Image
import streamlit as st
model = load_model('cat_dog_classifier.h5')

def get_image(image_file):
    return Image.open(image_file)
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256,256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return 'Dog'
    else:
        return 'Cat'

st.title("Cat Vs Dog Classifier")
img = st.file_uploader("Choose Image of cat or dog")

if st.button("Generate"):
    with st.spinner("Processing..."):
        image = get_image(img)
        result = predict_image(image)
        st.write(result)
