# human-action-classifier

This repository implements a robust machine learning pipeline for recognizing human actions using the KTH Dataset. The project focuses on handling class imbalance through balanced sampling and improving model accuracy using spatial-temporal feature extraction and data augmentation.

## Environment Setup
It is recommended to run this project within a virtual environment. Install the required dependencies using pip:

```bash
pip install numpy opencv-python scikit-learn torch matplotlib joblib notebook ipykernel
```

## Data Preparation
- Dataset Download: 2 options:
    - Option 1: Download the KTH dataset from [GSU Sharepoint](https://studentgsu-my.sharepoint.com/:u:/g/personal/ddo17_student_gsu_edu/IQAWdaMl6MXTTZ02_rtmUn6FAZBjVG3Xzca81dAxsE-59H0?e=M2jrwm) and unzip in this directory.
    - Option 2: Download the KTH database (including `00sequences.txt`, `walking.zip`, `jogging.zip`, `running.zip`, `boxing.zip`, `handwaving.zip`, `handclapping.zip`) from [KTH official website](https://www.csc.kth.se/cvap/actions/), extract, and place all folders and files in the `\archive` folder.

- Metadata & Sampling: Run data-info.py to parse the video metadata and perform balanced sampling to ensure equal representation across action classes.

- Data Splitting & Augmentation: Run data-split.py to crop the videos into individual action sequences. This script also applies Horizontal Flip augmentation to improve the model's robustness to directional changes. Processed clips and the resulting info.csv are saved in the data/ directory.

## Pipeline Execution
### The core logic for training and evaluation is handled within the Jupyter Notebook.
- Main Notebook: `classifier-refiner.ipynb`
- Workflow:
1. Feature Extraction: Detects interest points using Harris Corner Detector.
2. Codebook Construction: Creates a Visual Vocabulary using K-Means clustering ($K=800$).
3. Classification: Trains and evaluates multiple models to compare geometric and probabilistic approaches:
  - Gaussian Naive Bayes (GNB)
  - Multinomial Naive Bayes (MNB)
  - Linear Support Vector Machine (Linear SVM)
  - Multi-Layer Perceptron (MLP)

## Directory Structure
```
human-action-classifier/
├── archive/                  # Original KTH video files
├── data/                     # Processed video clips and info.csv
├── artifact/                 # Saved models (.joblib), descriptors (.pt), and logs
├── data-info.py              # Script for metadata parsing
├── data-split.py             # Script for cropping and augmentation
├── extract.py                # Core math and feature extraction module
└── classifier-refiner.ipynb  # Main training, refiner logic, and evaluation
```

## Results
This project incorporates an ablation study to validate the impact of each technique applied to the pipeline. The table below illustrates the step-by-step changes in accuracy resulting from data augmentation, the addition of the "Empty" sequence class, and the introduction of the Hierarchical Refiner strategy.

| Experimental Setup | Gaussian NB | Multinomial NB | Linear SVM | MLP |
| :--- | :---: | :---: | :---: | :---: |
| **Original (Baseline)** | 61.51% | 64.93% | 75.84% | 79.72% |
| **+ Empty Sequences** | 65.10% | 67.13% | 73.53% | 78.39% |
| **+ Transformation** | 69.01% | 71.32% | 77.88% | 83.02% |
| **+ Trans. & Empty** | 70.95% | 72.29% | 77.40% | 82.19% |
| **+ Trans., Empty & Refiner** | **72.74%** | **79.82%** | *71.47%* | **83.98%** |

### Key Analytical Insights

* **The Power of Data Augmentation**
  Applying spatial transformations, such as horizontal flipping, resulted in a significant performance boost across all four models (up to a 7% increase). This demonstrates that the models successfully learned the intrinsic motion characteristics of the actions rather than relying on directional noise.

* **Robustness via Empty Sequences**
  The introduction of the "Empty" background class improved the performance of probability-based models (Gaussian and Multinomial NB), though Linear SVM and MLP experienced a temporary dip. This dip reflects the readjustment of decision boundaries to accommodate a new feature space. Ultimately, this step was a crucial milestone for establishing model robustness in real-world environments without active human subjects.

* **Success of the Hierarchical Refiner**
  The implementation of the hierarchical refiner strategy stands out as a major success. Most notably, the Multinomial NB model saw a massive 7.5% point increase, jumping from 72.29% to 79.82%. By grouping overlapping locomotion actions (e.g., jogging and running) at the base level and delegating the fine-grained classification to a specialized refiner, the pipeline effectively resolved the primary classification bottleneck.

* **Deep Learning Capabilities vs. Geometric Limitations**
  The Multi-Layer Perceptron (MLP) achieved the highest overall accuracy at 83.98%. It successfully captured the complex, non-linear interactions within the high-dimensional ($K=800$) BoVW feature space. Conversely, the Linear SVM's accuracy dropped to the 71% range during the refiner stage. This highlights an interesting geometric limitation: as the subset of training data shrinks for the specialized refiner, the SVM struggles to find an optimal maximum-margin hyperplane, leading to slight overfitting.
