"""
Brain Tumor Detection - Model Training Script

This script trains the brain tumor classification model using AutoKeras.

Author: Aya Abdel Moniem
Email: Aya.abdelmoniem.afattah@gmail.com
Date: 2024
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import autokeras as ak
import tensorflow as tf

from data_preprocessing import DataPreprocessor


class BrainTumorTrainer:
    """
    Handles training of the brain tumor classification model.
    """
    
    def __init__(self, data_path, epochs=10, max_trials=5):
        """
        Initialize the trainer.
        
        Args:
            data_path (str): Path to dataset
            epochs (int): Number of training epochs
            max_trials (int): Number of AutoKeras trials
        """
        self.data_path = data_path
        self.epochs = epochs
        self.max_trials = max_trials
        self.model = None
        self.history = None
        
        # Initialize data preprocessor
        self.preprocessor = DataPreprocessor(data_path)
        
    def load_and_prepare_data(self):
        """
        Load and prepare training and testing data.
        
        Returns:
            tuple: (X_train, Y_train, X_test, Y_test)
        """
        print("=" * 60)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 60)
        
        # Load data
        X_train, Y_train = self.preprocessor.load_data('Training')
        X_test, Y_test = self.preprocessor.load_data('Testing')
        
        # Prepare data (shuffle and one-hot encode)
        X_train, Y_train = self.preprocessor.prepare_data(X_train, Y_train)
        X_test, Y_test = self.preprocessor.prepare_data(X_test, Y_test, shuffle_data=False)
        
        print(f"\nData shapes:")
        print(f"  Training: {X_train.shape}")
        print(f"  Testing: {X_test.shape}")
        
        return X_train, Y_train, X_test, Y_test
    
    def create_model(self):
        """
        Create AutoKeras ImageClassifier model.
        
        Returns:
            ak.ImageClassifier: AutoKeras model
        """
        print("\n" + "=" * 60)
        print("CREATING AUTOKERAS MODEL")
        print("=" * 60)
        
        clf = ak.ImageClassifier(
            overwrite=True,
            max_trials=self.max_trials,
            directory='autokeras_trials',
            project_name='brain_tumor_classification'
        )
        
        print(f"Model configuration:")
        print(f"  Max trials: {self.max_trials}")
        print(f"  Epochs per trial: {self.epochs}")
        
        return clf
    
    def train(self, X_train, Y_train, X_test, Y_test):
        """
        Train the model.
        
        Args:
            X_train: Training images
            Y_train: Training labels
            X_test: Testing images
            Y_test: Testing labels
        """
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        # Create model
        self.model = self.create_model()
        
        # Train model
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, Y_train,
            epochs=self.epochs,
            validation_data=(X_test, Y_test),
            verbose=1
        )
        
        print("\nTraining completed!")
    
    def evaluate(self, X_test, Y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test: Testing images
            Y_test: Testing labels
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)
        
        # Get predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(Y_test, axis=1)
        
        # Calculate accuracy
        test_loss, test_accuracy = self.model.evaluate(X_test, Y_test, verbose=0)
        
        print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate classification report
        class_names = self.preprocessor.get_class_names()
        report = classification_report(
            true_classes,
            predicted_classes,
            target_names=class_names,
            digits=4
        )
        
        print("\nClassification Report:")
        print(report)
        
        # Generate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'predictions': predictions,
            'confusion_matrix': cm,
            'report': report
        }
    
    def plot_confusion_matrix(self, cm, save_path='results/confusion_matrix.png'):
        """
        Plot and save confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        class_names = self.preprocessor.get_class_names()
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix - Brain Tumor Classification', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
        plt.close()
    
    def save_model(self, save_path='models/brain_tumor_model'):
        """
        Export and save the trained model.
        
        Args:
            save_path (str): Path to save the model
        """
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        
        # Export the best model
        exported_model = self.model.export_model()
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        exported_model.save(save_path, save_format='tf')
        
        print(f"\nModel saved to: {save_path}")
        
        # Print model summary
        print("\nModel Summary:")
        exported_model.summary()
        
        return exported_model


def main():
    """
    Main training pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classification Model')
    parser.add_argument('--data_path', type=str, default='Masoud- collection',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--max_trials', type=int, default=5,
                        help='Number of AutoKeras trials')
    parser.add_argument('--save_path', type=str, default='models/brain_tumor_model',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("BRAIN TUMOR CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Max trials: {args.max_trials}")
    print(f"  Save path: {args.save_path}")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize trainer
    trainer = BrainTumorTrainer(
        data_path=args.data_path,
        epochs=args.epochs,
        max_trials=args.max_trials
    )
    
    # Load and prepare data
    X_train, Y_train, X_test, Y_test = trainer.load_and_prepare_data()
    
    # Train model
    trainer.train(X_train, Y_train, X_test, Y_test)
    
    # Evaluate model
    results = trainer.evaluate(X_test, Y_test)
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(results['confusion_matrix'])
    
    # Save model
    model = trainer.save_model(args.save_path)
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFinal Test Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"\nModel saved to: {args.save_path}")
    print("Confusion matrix saved to: results/confusion_matrix.png")


if __name__ == "__main__":
    main()
