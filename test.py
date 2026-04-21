import pickle
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

# Load embeddings
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load query image
img = image.load_img('Screenshot 2026-03-15 085024.png', target_size=(224,224))
img_array = image.img_to_array(img)

expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Extract features
result = model.predict(preprocessed_img, verbose=0).flatten()
normalized_result = result / norm(result)

# KNN model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print("Similar images:")

for i in indices[0][1:6]:
    print(filenames[i])

    temp_img = cv2.imread(filenames[i])
    cv2.imshow("Recommended Image", cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)

cv2.destroyAllWindows()