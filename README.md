# Instagram User Classification and Like Count Prediction

This repository contains the implementation of machine learning models for Instagram user classification and like count prediction tasks.

## Repository Structure

### Core Components
- `common.py`: Contains shared utilities and data loading functions including:
  - Dataset loading and preprocessing
  - Text vectorization (TF-IDF)
  - Feature engineering for posts and user bios
  - PCA implementation for dimensionality reduction

### Classification Models
- `classification-svm.py`: Support Vector Machine classifier implementation
- `classification-randomforest.py`: Random Forest classifier implementation
- `classification-gradientboost.py`: Gradient Boosting classifier implementation

### Regression Model
- `regression-knn.py`: K-Nearest Neighbors regression model for like count prediction

## Methodology

### Text Processing and Feature Engineering
1. **Text Preprocessing**
   - Case folding for Turkish text
   - URL removal
   - Special character and punctuation removal
   - Number removal
   - Whitespace normalization

2. **Feature Extraction**
   - TF-IDF vectorization for both post captions and user bios
   - Separate feature spaces for posts (5000 features) and bios (500 features)
   - Weighted combination of post and bio features (posts: 1.0, bios: 0.2)

3. **Dimensionality Reduction**
   - PCA implementation for the Random Forest classifier
   - Feature scaling using StandardScaler

### Classification Approach
Multiple classifiers were implemented and compared:
1. **Support Vector Machine (SVM)**
   - Linear kernel
   - Various bio/caption weight combinations tested
   
2. **Random Forest**
   - PCA-reduced feature space
   - Balanced class weights
   - Optimized hyperparameters

3. **Gradient Boosting**
   - Default parameters
   - Potential for further optimization

## Experimental Results

### Classification Performance

#### SVM Experiments
| Configuration | Train Accuracy | Validation Accuracy |
|---------------|---------------|-------------------|
| Bio Weight 0.4 | 0.94 | 0.64 |
| Bio Weight 0.25 | 0.92 | 0.644 |
| Bio Weight 0.2 | 0.93 | 0.71 |
| Bio Weight 0.1 | 0.92 | 0.65 |

#### Random Forest Results
- Configuration: n_estimators=1000, max_depth=20
- Train Accuracy: 0.95
- Validation Accuracy: 0.57

#### Gradient Boost Results
- Default Parameters
- Train Accuracy: 0.99
- Validation Accuracy: 0.61

### Key Findings
1. SVM with bio weight of 0.2 achieved the best validation accuracy (0.71)
2. All models showed signs of overfitting, particularly Gradient Boost
3. Bio information provides valuable signal when properly weighted
4. Linear SVM outperformed more complex models in generalization

## Like Count Prediction
The regression task uses a KNN-based approach with:
- Advanced feature engineering for post metadata
- Ensemble of KNN predictions and historical averages
- Robust scaling for handling outliers

## Setup and Usage
1. Ensure all required libraries are installed (numpy, pandas, scikit-learn, nltk)
2. Run the desired classification model script
3. For like count prediction, run regression-knn.py
4. Results will be saved in the specified output files

## Required Libraries
- numpy
- pandas
- scikit-learn
- nltk
- scipy

## Author
Can Badem (can.badem)
