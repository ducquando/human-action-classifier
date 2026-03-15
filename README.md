# human-action-classifier
## [Project Title] KTH Action Recognition Pipeline

## Project Overview
### This repository implements a robust machine learning pipeline for recognizing human actions using the KTH Dataset. The project focuses on handling class imbalance through balanced sampling and improving model accuracy using spatial-temporal feature extraction and data augmentation.

## Environment Setup

## Data Preparation

## Pipeline Execution
### The core logic for training and evaluation is handled within the Jupyter Notebook.
- Main Notebook: schudlt copy.ipynb
- Workflow:
1. Feature Extraction: Detects interest points using Harris Corner Detector.
2. Codebook Construction: Creates a Visual Vocabulary using K-Means clustering ($K=800$).
3. Classification: Trains and evaluates Gaussian NB, Multinomial NB, and Linear SVM.

## Directory Structure
- data-info.py: Script for metadata parsing and balanced sampling.
- data-split.py: Script for video cropping and augmentation.
- schudlt copy.ipynb: Main training and evaluation notebook.
- archive/: Original KTH video files.
- data/: Processed video clips and info.csv.
- artifact/: Saved .joblib models and result logs.

## Results