"""
Brain Tumor Detection - Data Preprocessing Module

This module handles loading, preprocessing, and augmentation of brain MRI images.

Author: Aya Abdel Moniem
Email: Aya.abdelmoniem.afattah@gmail.com
Date: 2024
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


class DataPreprocessor:
    """
    Handles all data preprocessing operations including loading,
    filtering, color mapping, and augmentation.
    """
    
    def __init__(self, data_path, img_size=(200, 200)):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path (str): Path to the dataset directory
            img_size (tuple): Target image size (width, height)
        """
        self.data_path = data_path
        self.img_size = img_size
        self.classes = {
            'notumor': 0,
            'pituitary': 1,
            'meningioma': 2,
            'glioma': 3
        }
        
    def preprocess_image(self, image_path):
        """
        Apply preprocessing pipeline to a single image.
        
        Steps:
        1. Load image in grayscale
        2. Apply bilateral filter for noise reduction
        3. Apply color map for better visualization
        4. Resize to target size
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Load image in grayscale
        image = cv2.imread(image_path, 0)
        
        # Apply bilateral filter to remove noise while preserving edges
        # Parameters: diameter=2, sigmaColor=50, sigmaSpace=50
        image = cv2.bilateralFilter(image, 2, 50, 50)
        
        # Apply color map (BONE colormap creates a pseudocolored image)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        
        # Resize to target size
        image = cv2.resize(image, self.img_size)
        
        return image
    
    def load_data(self, data_type='Training'):
        """
        Load and preprocess all images from the dataset.
        
        Args:
            data_type (str): 'Training' or 'Testing'
            
        Returns:
            tuple: (X, Y) where X is images array and Y is labels array
        """
        X = []
        Y = []
        
        print(f"Loading {data_type} data...")
        
        for cls in self.classes:
            # Construct path to class folder
            class_path = os.path.join(self.data_path, data_type, cls)
            
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist!")
                continue
            
            # Process all images in the class folder
            for file in tqdm(os.listdir(class_path), desc=f"Processing {cls}"):
                try:
                    image_path = os.path.join(class_path, file)
                    image = self.preprocess_image(image_path)
                    X.append(image)
                    Y.append(self.classes[cls])
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    continue
        
        return np.array(X), np.array(Y)
    
    def prepare_data(self, X, Y, shuffle_data=True):
        """
        Prepare data for training by shuffling and one-hot encoding.
        
        Args:
            X (np.ndarray): Images array
            Y (np.ndarray): Labels array
            shuffle_data (bool): Whether to shuffle the data
            
        Returns:
            tuple: (X, Y) preprocessed and ready for training
        """
        # Shuffle data if requested
        if shuffle_data:
            X, Y = shuffle(X, Y, random_state=42)
        
        # One-hot encode labels
        Y = keras.utils.to_categorical(Y, num_classes=len(self.classes))
        
        return X, Y
    
    def create_data_augmentation(self):
        """
        Create ImageDataGenerator for data augmentation.
        
        Augmentation techniques:
        - Random rotation (±10 degrees)
        - Width shift (5%)
        - Height shift (5%)
        - Horizontal flip
        
        Returns:
            ImageDataGenerator: Configured data augmentation generator
        """
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True
        )
        
        return datagen
    
    def get_class_names(self):
        """
        Get list of class names in order.
        
        Returns:
            list: Class names
        """
        return list(self.classes.keys())
    
    def get_num_classes(self):
        """
        Get number of classes.
        
        Returns:
            int: Number of classes
        """
        return len(self.classes)


def main():
    """
    Example usage of the DataPreprocessor class.
    """
    # Initialize preprocessor
    data_path = "Masoud- collection"
    preprocessor = DataPreprocessor(data_path)
    
    # Load training data
    X_train, Y_train = preprocessor.load_data('Training')
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {Y_train.shape}")
    
    # Load testing data
    X_test, Y_test = preprocessor.load_data('Testing')
    print(f"Testing data shape: {X_test.shape}")
    print(f"Testing labels shape: {Y_test.shape}")
    
    # Prepare data
    X_train, Y_train = preprocessor.prepare_data(X_train, Y_train)
    X_test, Y_test = preprocessor.prepare_data(X_test, Y_test, shuffle_data=False)
    
    print(f"\nFinal training data shape: {X_train.shape}")
    print(f"Final training labels shape: {Y_train.shape}")
    print(f"Final testing data shape: {X_test.shape}")
    print(f"Final testing labels shape: {Y_test.shape}")
    
    # Create data augmentation
    datagen = preprocessor.create_data_augmentation()
    datagen.fit(X_train)
    
    print("\nData preprocessing completed successfully!")


if __name__ == "__main__":
    main()
