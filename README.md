# Personality Prediction

This repository contains a solution for the [Kaggle Playground Series - Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7/data) competition.

## Project Goal

The objective of this project is to build a binary classification model to predict whether a person is an Introvert or an Extrovert based on their social behavior and personality traits provided in the dataset.

## Model & Approach

This solution uses an **XGBoost (Extreme Gradient Boosting)** classifier, a powerful and popular algorithm for tabular data, to perform the classification.

The main notebook (located in the `/Notebooks` folder) covers the following steps:
* **Exploratory Data Analysis (EDA):** Understanding the features and their relationships with the target variable (`Personality`).
* **Feature Engineering:** Creating new features to improve model performance.
* **Model Training:** Training the XGBoost model on the dataset.
* **Hyperparameter Tuning:** Optimizing the model's parameters for the best accuracy.
* **Submission:** Generating the final `submission.csv` file.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RaymussenArthur/asd.git](https://github.com/RaymussenArthur/asd.git)
    cd asd
    ```

2.  **Install dependencies:**
    (It's highly recommended to create and add a `requirements.txt` file)
    ```bash
    pip install pandas numpy xgboost scikit-learn matplotlib seaborn
    ```

3.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
4.  Open and run the notebook located inside the `/Notebooks` directory.
