# Stock News Sentiment Analysis

## 🎯 Project Overview

This project aims to leverage Natural Language Processing (NLP) and Machine Learning techniques to analyze the sentiment embedded in financial news headlines and predict corresponding stock price movements. The core goal is to develop **sophisticated tools to analyze market sentiment** and **integrate this information into investment strategies**.

By quantifying the sentiment of daily news, this system provides a powerful indicator to complement traditional technical and fundamental analysis in stock trading.

## ✨ Features & Methodology

The project follows a rigorous data science pipeline, from data preparation and exploratory analysis to advanced model training and evaluation.

### 1. Data Processing and Exploration (EDA)
* **Data Structure:** The dataset includes daily data points with financial news and corresponding stock metrics: `Date`, `News` (text content), `Open`, `High`, `Low`, `Close`, `Volume`, and a `Label` (sentiment/stock movement indicator).
* **Data Cleaning:** The initial `Date` column, which was of `object` type, was converted to the `datetime` datatype for time-series analysis.
* **Univariate Analysis:** Extensive Exploratory Data Analysis (EDA) was performed, including univariate analysis for all stock price columns (`Open`, `High`, `Low`, `Close`), trading `Volume`, sentiment `Label`, and the distribution of the news content length. Visualization tools like **Seaborn** were used to generate histograms and box plots for key insights.

### 2. Text Representation (Embeddings)

To prepare the news text for machine learning, two distinct embedding techniques were employed:
* **Word2Vec:** A classic approach to generate dense vector representations of the news articles.
* **Sentence Transformer:** A more modern, BERT-based technique used to create high-quality sentence-level embeddings, capturing richer semantic meaning.

### 3. Predictive Modeling

Multiple classification models were implemented and rigorously compared:
* **Traditional Machine Learning:**
    * **Random Forest Classifier** (`RandomForestClassifier`).
* **Deep Learning Models:**
    * An **DNN (Deep Neural Network)** network was trained, confirming the use of a deep learning architecture. The training process showed successful learning across multiple epochs with decreasing loss and increasing validation accuracy.

## 📊 Key Results

The performance of various models was evaluated based on standard classification metrics. The final evaluation metrics suggest the robust predictive power of the combined NLP and DL approach with Sentence Transformers:

| Metric | Best Performance |
| :--- | :--- |
| **Test Accuracy** | ~80.28% |
| **Test Precision** | ~79.05% |
| **Test Recall** | ~80.28% |
| **Test F1-Score** | ~79.66% |

A comparative analysis was performed across different Random Forest models using both Word2Vec and Sentence Transformer embeddings, documenting the impact of hyperparameters like `n_estimators` and `max_depth` for ML models and `epochs`, `number of layers` and `batch size` for DL models on performance.

## 🛠 Technologies & Dependencies

This project was built primarily using the Python programming language and its scientific computing stack.

* **Language:** Python
* **Data Manipulation:** `pandas`
* **Numerical Computing:** `numpy`
* **Visualization:** `seaborn`, `matplotlib`
* **Machine Learning/NLP:**
    * `scikit-learn` (for Random Forest and evaluation metrics)
    * `Word2Vec` (for word embeddings)
    * `Sentence Transformer` (for advanced sentence embeddings)
    * `TensorFlow` / `Keras` (for implementing DNN)
