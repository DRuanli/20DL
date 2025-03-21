#!/usr/bin/env python3
"""
Main script for Sentiment Analysis Assignment
TDTU 503077 - DEEP LEARNING

This script orchestrates the entire workflow:
1. Environment setup verification
2. Data generation/validation
3. Model training and evaluation
4. Results visualization

Run this script to execute the complete pipeline.
"""

import os
import sys
import subprocess
import time
import json
import pandas as pd
from pathlib import Path


def print_colored(text, color="green"):
    """Print colored text to the terminal"""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "end": "\033[0m"
    }

    print(f"{colors.get(color, '')}{text}{colors['end']}")


def print_section(title):
    """Print a section header"""
    line = "=" * 80
    print("\n")
    print_colored(line, "blue")
    print_colored(f"{title.center(80)}", "bold")
    print_colored(line, "blue")
    print("")


def run_command(command):
    """Run a shell command and stream output in real-time"""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Print stdout in real-time
    for line in process.stdout:
        print(line, end='')

    # Get return code
    process.wait()

    # Check if there were any errors
    if process.returncode != 0:
        print_colored("\nError running command:", "red")
        print_colored(f"$ {command}", "yellow")
        for line in process.stderr:
            print_colored(line, "red")
        return False

    return True


def check_environment():
    """Verify that all required packages are installed"""
    print_section("Checking Environment")

    # Check Python version
    print("Python version:", sys.version.split()[0])
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print_colored("Error: Python 3.8 or higher is required", "red")
        print_colored(f"Current version: {python_version.major}.{python_version.minor}", "red")
        sys.exit(1)

    # Required packages
    required_packages = [
        "torch", "torchtext", "pandas", "numpy",
        "nltk", "scikit-learn", "matplotlib"
    ]

    # Check each package
    missing_packages = []
    print("\nChecking required packages:")

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print_colored(f"✗ {package} (missing)", "red")

    # Handle missing packages
    if missing_packages:
        print_colored("\nSome required packages are missing.", "red")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        print(f"To install them, run: {install_cmd}")

        response = input("\nWould you like to install them now? (y/n): ")
        if response.lower() == 'y':
            print_section(f"Installing Missing Packages")
            if not run_command(install_cmd):
                print_colored("Failed to install packages. Please install them manually.", "red")
                sys.exit(1)
        else:
            print_colored("Please install the missing packages and try again.", "red")
            sys.exit(1)

    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
        print("\nCreated 'output' directory for saving results")

    print_colored("\nEnvironment check complete! ✓", "green")


def prepare_data():
    """Prepare the dataset for training"""
    print_section("Data Preparation")

    data_file = Path("sentiment_data.csv")

    if data_file.exists():
        print(f"Found existing dataset: {data_file}")

        # Read and validate the data
        try:
            df = pd.read_csv(data_file)
            sample_count = len(df)
            label_counts = df['label'].value_counts()

            print(f"Dataset contains {sample_count} samples")
            print("\nLabel distribution:")
            for label, count in label_counts.items():
                percentage = (count / sample_count) * 100
                print(f"  {label}: {count} samples ({percentage:.1f}%)")

            # Check if dataset meets requirements
            if sample_count < 500:
                print_colored(f"\nWarning: Dataset contains only {sample_count} samples", "yellow")
                print_colored("The assignment requires at least 500 samples", "yellow")
                regenerate = True
            else:
                print_colored("\nDataset meets the minimum size requirement ✓", "green")

                # Ask if user wants to regenerate anyway
                response = input("\nWould you like to regenerate the dataset? (y/n): ")
                regenerate = response.lower() == 'y'

        except Exception as e:
            print_colored(f"Error reading dataset: {e}", "red")
            print_colored("Will regenerate the dataset", "yellow")
            regenerate = True
    else:
        print("No existing dataset found")
        regenerate = True

    # Generate the dataset if needed
    if regenerate:
        print_section("Generating Dataset")
        print("Creating sentiment dataset with at least 500 samples...")

        if not run_command("python generate_data.py"):
            print_colored("Failed to generate dataset", "red")
            sys.exit(1)

        # Verify the newly generated dataset
        try:
            df = pd.read_csv(data_file)
            print_colored(f"\nSuccessfully generated dataset with {len(df)} samples ✓", "green")
        except Exception as e:
            print_colored(f"Error reading generated dataset: {e}", "red")
            sys.exit(1)


def run_experiments():
    """Run the sentiment analysis model training and evaluation"""
    print_section("Running Experiments")

    print("This will run two experiments:")
    print("1. RNN with pretrained GloVe embeddings")
    print("2. RNN with embeddings trained from scratch")
    print("\nThe process may take several minutes to complete.")

    # Ask for confirmation before starting
    print("\nExperiments are ready to start.")
    input("Press Enter to continue...")

    start_time = time.time()

    # Run the training and evaluation script
    if not run_command("python train_eval.py"):
        print_colored("Error running experiments", "red")
        sys.exit(1)

    # Calculate total runtime
    runtime = time.time() - start_time
    minutes, seconds = divmod(runtime, 60)

    print_colored(f"\nExperiments completed in {int(minutes)}m {int(seconds)}s ✓", "green")


def display_results():
    """Display the results of the experiments"""
    print_section("Results Summary")

    # Check if results file exists
    results_file = Path("results.json")
    if not results_file.exists():
        print_colored("Results file not found", "red")
        print("Make sure the experiments completed successfully")
        return False

    # Load and display results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Extract metrics
        pre_acc = results["RNN_Pretrained=True"]["Accuracy"]
        pre_f1 = results["RNN_Pretrained=True"]["F1-score"]
        scratch_acc = results["RNN_Pretrained=False"]["Accuracy"]
        scratch_f1 = results["RNN_Pretrained=False"]["F1-score"]

        # Display results table
        print_colored("Model Performance Comparison:", "bold")
        print_colored("┌─────────────────┬────────────┬────────────┐", "cyan")
        print_colored("│ Model           │ Accuracy   │ F1-score   │", "cyan")
        print_colored("├─────────────────┼────────────┼────────────┤", "cyan")
        print_colored(f"│ Pretrained GloVe │ {pre_acc:.4f}    │ {pre_f1:.4f}    │", "cyan")
        print_colored(f"│ Scratch Embeddings │ {scratch_acc:.4f}    │ {scratch_f1:.4f}    │", "cyan")
        print_colored("└─────────────────┴────────────┴────────────┘", "cyan")

        # Display observation
        if "observation" in results:
            print("\nObservation:")
            print_colored(f"  {results['observation']}", "yellow")

        # Check for plots
        plot_files = list(Path("output").glob("*.png"))
        if plot_files:
            print("\nGenerated plots:")
            for plot in plot_files:
                print(f"  {plot}")

        return True

    except Exception as e:
        print_colored(f"Error displaying results: {e}", "red")
        return False


def next_steps():
    """Display next steps for completing the assignment"""
    print_section("Next Steps for Assignment Completion")

    print("1. Complete your PDF report with the following sections:")
    print("   - Theory: Explanation of word embeddings and RNNs")
    print("   - Description: Details of your data preparation and model architecture")
    print("   - Results: Analysis of experimental outcomes")
    print("   - Conclusion: Summary of findings and potential improvements")

    print("\n2. Ensure you submit the following files:")
    print("   - PDF report")
    print("   - Code files (data.py, model.py, train_eval.py)")
    print("   - Dataset (sentiment_data.csv)")
    print("   - Results (results.json)")

    print("\n3. Submission deadline: March 28, 2025 (11:59 PM)")

    print_colored("\nGood luck with your assignment! 🎓", "green")


def main():
    """Main execution function"""
    try:
        print_colored("\n===== SENTIMENT ANALYSIS ASSIGNMENT =====", "purple")
        print_colored("TDTU 503077 - DEEP LEARNING", "purple")
        print("")

        # Step 1: Check environment
        check_environment()

        # Step 2: Prepare data
        prepare_data()

        # Step 3: Run experiments
        run_experiments()

        # Step 4: Display results
        display_results()

        # Step 5: Show next steps
        next_steps()

    except KeyboardInterrupt:
        print_colored("\nProcess interrupted by user", "yellow")
        sys.exit(1)
    except Exception as e:
        print_colored(f"\nUnexpected error: {e}", "red")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()