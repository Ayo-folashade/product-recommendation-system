import streamlit as st
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
import pickle

# Load the pre-trained ResNet50 model
model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the model's weights, so they won't be updated during training
model.trainable = False

# Create a new model that appends a GlobalMaxPooling2D layer after the ResNet50 model
model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalMaxPooling2D()
])


# Function to extract features from an image using the pre-trained model
def extract_features(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.resnet50.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


# Load the pickled embeddings and filenames
embeddings = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))


# Function to compute similarity between the query image and the database images
def compute_similarity(query_embedding, database_embeddings):
    # Compute cosine similarity between the query and all database embeddings
    similarities = np.dot(query_embedding, np.array(database_embeddings).T)
    # Get the indices of the top k most similar images
    k = 5  # You can choose how many similar images to show
    top_indices = np.argsort(similarities)[::-1][:k]
    return top_indices


# Streamlit app code
def main():
    st.title("Product Recommendation")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Preprocess the user-uploaded image
        query_embedding = extract_features(uploaded_image, model)

        # Compute similar images
        similar_indices = compute_similarity(query_embedding, embeddings)

        st.subheader("Uploaded Image")
        st.image(uploaded_image)

        st.subheader("Products Available")
        if len(similar_indices) == 0:
            st.write("No similar products available.")
        else:
            for idx in similar_indices:
                if idx < len(filenames):
                    st.image(filenames[idx])


if __name__ == "__main__":
    main()
