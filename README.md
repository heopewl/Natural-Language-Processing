# Natural Language Processing - Bird Species Classification

## Project Overview
This project aims to classify bird species based on their descriptions using Natural Language Processing (NLP) and Machine Learning techniques. The dataset focuses on four specific species: Black-naped Oriole, Javan Myna, Little Egret, and Collared Kingfisher. The project is divided into three main parts: data understanding and pre-processing, model training and selection, and final evaluation on a test set.

## Project Structure
- `NLP Part_1.ipynb`: Data exploration, visualization, and pre-processing.
- `NLP Part_2.ipynb`: Feature extraction using TF-IDF and comparison of various machine learning models (Decision Tree, SVM, Naive Bayes, Logistic Regression).
- `NLP Part_3.ipynb`: Final model evaluation and performance analysis on the test dataset.
- `Data/`: Contains the raw and cleaned datasets.
- `Models/`: Contains the saved best-performing model (SVM) and the TF-IDF vectorizer.

## Methodology

### 1. Data Pre-processing
The raw data was cleaned and transformed to improve model performance:
- **Handling Missing Values & Duplicates**: Removed rows with missing descriptions and duplicate entries.
- **Text Cleaning**: Converted text to lowercase, removed special characters, and removed numbers.
- **Normalization**: Replaced month abbreviations with full forms and removed generic stop words.
- **Lemmatization**: Used SpaCy's English model for lemmatizing words to their base forms.
- **Filtering**: Removed words with fewer than 3 characters.

### 2. Feature Extraction
A **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** was used to convert text data into numerical vectors.
- **N-Grams**: Range set to (1, 2) to capture both individual words and word pairs.
- **Result**: Generated a feature space of 8,940 unique tokens from the training data.

### 3. Model Training & Selection
Four models were compared using 80/20 train-test split:
- Logistic Regression
- Naive Bayes
- Decision Tree
- **Support Vector Machine (SVM)** - *Selected as the best model.*

**Comparison Table (Validation Set):**
| Metric | Decision Tree | SVM | Naive Bayes | Logistic Regression |
| :--- | :--- | :--- | :--- | :--- |
| Accuracy | 0.888 | **0.932** | 0.915 | 0.923 |
| Precision | 0.881 | **0.936** | 0.931 | 0.926 |
| Recall | 0.881 | **0.923** | 0.888 | 0.912 |
| F1-Score | 0.879 | **0.929** | 0.902 | 0.918 |

## Final Results (Test Set)
The selected SVM model demonstrated high accuracy and robustness on the unseen test data:
- **Overall Accuracy**: 97.37% (37/38 correctly predicted)
- **Macro Avg Precision**: 0.96
- **Macro Avg Recall**: 0.98
- **Macro Avg F1-Score**: 0.97

**Per-Species Performance:**
| Species | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Black-naped Oriole | 1.00 | 0.94 | 0.97 |
| Collared Kingfisher | 1.00 | 1.00 | 1.00 |
| Javan Myna | 1.00 | 1.00 | 1.00 |
| Little Egret | 0.86 | 1.00 | 0.92 |

## Future Improvements
- **Data Balancing**: Address class imbalance (specifically for Little Egret) using oversampling techniques like SMOTE.
- **Data Augmentation**: Increase training samples for species with lower precision/recall.
- **Hyperparameter Tuning**: Optimize model parameters for even better performance.
