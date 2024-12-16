# product_finder/image_analysis.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import logging

class ImageAnalyzer:
    def __init__(self):
        # Load pre-trained ResNet50 model, excluding the top classification layer
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
    def extract_features(self, img_path):
        """Extract features from an image using ResNet50."""
        try:
            # Load and preprocess the image
            if isinstance(img_path, str):
                img = image.load_img(img_path, target_size=(224, 224))
            else:
                img = Image.open(img_path)
                img = img.resize((224, 224))
            
            # Convert image to array and preprocess
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features using the model
            features = self.model.predict(img_array)
            return features.flatten()
        except Exception as e:
            logging.error(f"Error in feature extraction: {str(e)}")
            raise Exception(f"Failed to analyze image: {str(e)}")
    
    def compare_features(self, features1, features2):
        """Compare two feature vectors using cosine similarity."""
        try:
            similarity = np.dot(features1, features2) / (
                np.linalg.norm(features1) * np.linalg.norm(features2)
            )
            return similarity
        except Exception as e:
            logging.error(f"Error in feature comparison: {str(e)}")
            raise Exception(f"Failed to compare images: {str(e)}")