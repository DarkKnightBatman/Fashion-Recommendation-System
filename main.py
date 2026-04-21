import streamlit as st
import os
import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2


st.title("Fashion Recommender System")


# ---------------------------
# Load embeddings and filenames
# ---------------------------

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


# ---------------------------
# Load model
# ---------------------------

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


# ---------------------------
# Save uploaded file
# ---------------------------

def save_uploaded_file(uploaded_file):

    try:

        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        file_path = os.path.join("uploads", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    except:
        return None


# ---------------------------
# Feature extraction
# ---------------------------

def feature_extraction(img_path, model):

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)

    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


# ---------------------------
# Recommendation function
# ---------------------------

def recommend(features, feature_list):

    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


# ---------------------------
# Streamlit UI
# ---------------------------

uploaded_file = st.file_uploader("Choose an Image")


if uploaded_file is not None:

    file_path = save_uploaded_file(uploaded_file)

    if file_path is not None:

        display_image = cv2.imread(file_path)
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        st.image(display_image)

        features = feature_extraction(file_path, model)

        indices = recommend(features, feature_list)

        st.subheader("Recommended Items")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][1]])

        with col2:
            st.image(filenames[indices[0][2]])

        with col3:
            st.image(filenames[indices[0][3]])

        with col4:
            st.image(filenames[indices[0][4]])

        with col5:
            st.image(filenames[indices[0][5]])

    else:

        st.error("Some error occurred while uploading the file")