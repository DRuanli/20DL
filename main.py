"""
Sentiment Analysis Assignment Runner
This script runs the entire pipeline for the sentiment analysis assignment:
1. Generate the dataset
2. Process the data
3. Train and evaluate models
4. Print results
"""

import os
import subprocess
import sys

def check_environment():
    """Check if required libraries are installed"""
    try:
        import torch
        import pandas
        import nltk
        import sklearn
        print("✓ All required libraries are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing library: {e}")
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return False

def run_data_generation():
    """Run the data generation script"""
    print("\n=== GENERATING DATA ===")
    try:
        # Check if dataset already exists
        if os.path.exists("sentiment_data.csv"):
            choice = input("Dataset already exists. Regenerate? (y/n): ")
            if choice.lower() != 'y':
                print("Using existing dataset.")
                return True

        # Run the data generation script
        from data_generator import generate_sentiment_dataset
        sentiment_df = generate_sentiment_dataset(550)
        sentiment_df = sentiment_df.dropna()
        if len(sentiment_df) >= 500:
            sentiment_df = sentiment_df.iloc[:500]
            sentiment_df.to_csv('sentiment_data.csv', index=False)
            print(f"✓ Successfully generated dataset with {len(sentiment_df)} samples")
            return True
        else:
            print(f"✗ Could only generate {len(sentiment_df)} valid samples, need at least 500")
            return False
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        return False

def run_model_training():
    """Run the model training and evaluation"""
    print("\n=== TRAINING MODELS ===")
    try:
        # Import and run the training script
        import train_eval
        print("✓ Model training and evaluation completed")
        return True
    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return False

def main():
    """Main function to run the entire pipeline"""
    print("=== SENTIMENT ANALYSIS ASSIGNMENT RUNNER ===")

    # Step 1: Check environment
    if not check_environment():
        print("Please restart the script after installing requirements")
        return

    # Step 2: Generate data
    if not run_data_generation():
        print("Data generation failed or insufficient samples")
        return

    # Step 3: Run model training and evaluation
    if not run_model_training():
        print("Model training failed")
        return

    # Step 4: Final instructions
    print("\n=== ASSIGNMENT COMPLETED SUCCESSFULLY ===")
    print("1. Check results.json for model performance metrics")
    print("2. Complete the report template in 'Assignment 2 Report.md'")
    print("3. Submit your files as specified in the assignment")

if __name__ == "__main__":
    main()