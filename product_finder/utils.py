import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from PIL import Image
from io import BytesIO
import logging

class ImageAnalyzer:
    def __init__(self):
        # Load pre-trained ResNet50 model, excluding the top classification layer
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
    def extract_features(self, img_path):
        """Extract features from an image using ResNet50."""
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
    
    def compare_features(self, features1, features2):
        """Compare two feature vectors using cosine similarity."""
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )
        return similarity

class ProductScraper:
    def __init__(self):
        # Configure Chrome options for headless operation
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=chrome_options)
        self.image_analyzer = ImageAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    def get_product_categories(self, image_features):
        """Use image features to determine relevant product categories."""
        # Here you could implement category classification
        # For now, we'll use some general product categories
        return ['electronics', 'clothing', 'accessories', 'home']
    
    def search_amazon(self, image_features, categories):
        """Search Amazon for visually similar products."""
        products = []
        
        for category in categories:
            try:
                url = f"https://www.amazon.in/s?k={category}"
                self.driver.get(url)
                
                # Wait for product elements to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.s-result-item'))
                )
                
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                for item in soup.select('.s-result-item'):
                    try:
                        # Extract product information
                        title_elem = item.select_one('.a-text-normal')
                        price_elem = item.select_one('.a-price-whole')
                        img_elem = item.select_one('img.s-image')
                        
                        if not all([title_elem, price_elem, img_elem]):
                            continue
                            
                        title = title_elem.text
                        price = price_elem.text
                        img_url = img_elem['src']
                        product_url = 'https://amazon.in' + item.select_one('a.a-link-normal')['href']
                        
                        # Download and analyze product image
                        response = requests.get(img_url)
                        img_content = BytesIO(response.content)
                        product_features = self.image_analyzer.extract_features(img_content)
                        
                        # Calculate similarity with uploaded image
                        similarity = self.image_analyzer.compare_features(
                            image_features, product_features
                        )
                        
                        if similarity > 0.5:  # Only include products with significant similarity
                            products.append({
                                'title': title,
                                'price': float(price.replace(',', '')),
                                'url': product_url,
                                'image_url': img_url,
                                'similarity': similarity
                            })
                    
                    except Exception as e:
                        self.logger.error(f"Error processing product: {str(e)}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error searching Amazon category {category}: {str(e)}")
                continue
                
        return products
    
    def search_flipkart(self, image_features, categories):
        """Search Flipkart for visually similar products."""
        products = []
        
        for category in categories:
            try:
                url = f"https://www.flipkart.com/search?q={category}"
                self.driver.get(url)
                
                # Wait for product elements to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '._1AtVbE'))
                )
                
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                for item in soup.select('._1AtVbE'):
                    try:
                        # Extract product information
                        title_elem = item.select_one('._4rR01T')
                        price_elem = item.select_one('._30jeq3')
                        img_elem = item.select_one('._396cs4')
                        
                        if not all([title_elem, price_elem, img_elem]):
                            continue
                            
                        title = title_elem.text
                        price = price_elem.text[1:]  # Remove ₹ symbol
                        img_url = img_elem['src']
                        product_url = 'https://flipkart.com' + item.select_one('._1fQZEK')['href']
                        
                        # Download and analyze product image
                        response = requests.get(img_url)
                        img_content = BytesIO(response.content)
                        product_features = self.image_analyzer.extract_features(img_content)
                        
                        # Calculate similarity with uploaded image
                        similarity = self.image_analyzer.compare_features(
                            image_features, product_features
                        )
                        
                        if similarity > 0.5:  # Only include products with significant similarity
                            products.append({
                                'title': title,
                                'price': float(price.replace(',', '')),
                                'url': product_url,
                                'image_url': img_url,
                                'similarity': similarity
                            })
                    
                    except Exception as e:
                        self.logger.error(f"Error processing product: {str(e)}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error searching Flipkart category {category}: {str(e)}")
                continue
                
        return products

    
    def compare_images(self, hash1, hash2):
        """Compare two image hashes and return similarity score."""
        return 1 - (bin(int(hash1, 16) ^ int(hash2, 16)).count('1') / 64)
