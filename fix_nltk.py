"""
NLTK Resource Downloader
This script ensures all necessary NLTK resources are downloaded.
Run this script before running the sentiment analysis pipeline.
"""

import nltk
import sys


def download_nltk_resources():
    """Download all required NLTK resources for the sentiment analysis project"""

    print("Downloading NLTK resources...")

    # List of resources needed
    resources = [
        'punkt',
        'perluniprops',
        'nonbreaking_prefixes',
        'crubadan',
        'zhai_bolton'
    ]

    # Download each resource
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"✓ {resource} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {resource}: {e}")

    print("\nVerifying NLTK installations...")

    # Try tokenizing a sample sentence to verify installation
    test_sentence = "This is a test sentence for NLTK tokenization."
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(test_sentence)
        print(f"Tokenization test successful: {tokens}")
        print("\nAll NLTK resources installed successfully!")
        return True
    except Exception as e:
        print(f"Tokenization test failed: {e}")
        print("\nThere may be issues with NLTK installation.")
        return False


if __name__ == "__main__":
    download_nltk_resources()