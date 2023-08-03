# Image-Based Recommendation System using ResNet-50 and Cosine Similarity

## Overview

This GitHub repository contains the code for an Image-Based Recommendation System that utilizes the ResNet-50 deep learning model as a feature extractor and employs cosine similarity to find the top ```K``` similar embeddings for a given input image. 

The system aims to provide users with personalized recommendations based on the visual content of the image provided.

## Features
- Utilizes the powerful ResNet-50 model pre-trained on the imagenet dataset for feature extraction.
- Employs cosine distance as the similarity metric for finding similar embeddings.
- Returns the top ```K``` most similar images based on the cosine similarity score.
- Easy-to-use and adaptable codebase.

## Requirements

- ```Python 3.x```
- ```tensorflow ==2.5.0```
- ```numpy==1.21.0```
- ```streamlit```
- ```pillow==8.3.1```
- ```pandas```

## Usage
- Prepare your dataset: Ensure you have a dataset of images that you want to use for recommendations. The dataset should be organized in an appropriate folder structure.
- The dataset used: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

- Feature Extraction: The first step is to use the pre-trained ResNet-50 model to extract embeddings from your dataset. Run ```new.py``` to extract the embeddings and save the pickle files

- Recommendation: Now, you can use the generated embeddings to get recommendations for a specific image. Run ```main.py```. The script will display the top ```K``` images that are most similar to the query image based on cosine similarity.


## Acknowledgments
The image recommendation system is built upon the incredible work done by the Tensorflow, Numpy, Pandas and Streamlit, Kaggle communities, as well as the original authors of the ResNet-50 model.
