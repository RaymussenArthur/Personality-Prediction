# Personality Prediction (Kaggle S5E7)

This repository contains the solution for the [Kaggle Playground Series - Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7) competition. The goal is to predict a person's 16-type personality based on their answers to a custom survey.

This project uses an **XGBoost Classifier** to build four independent binary models—one for each of the four personality dichotomies—to achieve a robust and accurate prediction.

---

## 🚀 Project Overview

The 16 personality types (like INTJ, ESFP, etc.) are a combination of four binary traits:

1.  **Introversion (I)** vs. **Extroversion (E)**
2.  **Intuition (N)** vs. **Sensing (S)**
3.  **Feeling (F)** vs. **Thinking (T)**
4.  **Perceiving (P)** vs. **Judging (J)**

Instead of treating this as a complex 16-class classification problem, this solution builds **four separate binary classification models**. The final personality type is determined by concatenating the results of these four models (e.g., `I` + `N` + `T` + `J` = `INTJ`).

---

## 🔧 Methodology & Pipeline

### 1. Data Preprocessing

* **Loading:** The `train.csv` and `test.csv` files are loaded using `pandas`.
* **Target Engineering:** The single `Personality` column (e.g., 'INFP') in the training data is split into **four separate binary target variables**: `is_I`, `is_N`, `is_F`, and `is_P`.
    * `'INFP'` -> `is_I=1`, `is_N=1`, `is_F=1`, `is_P=1`
    * `'ESTJ'` -> `is_I=0`, `is_N=0`, `is_F=0`, `is_P=0`
* **Features:** The survey questions serve as the features (`X`). The `id` column is dropped.

### 2. Modeling with XGBoost

* An `XGBClassifier` is used as the base model for its high performance and speed.
* **Four models** are trained independently:
    1.  `model_IE`: Predicts `is_I` (Introversion/Extroversion) using `X`
    2.  `model_NS`: Predicts `is_N` (Intuition/Sensing) using `X`
    3.  `model_FT`: Predicts `is_F` (Feeling/Thinking) using `X`
    4.  `model_PJ`: Predicts `is_P` (Perceiving/Judging) using `X`
* The `Modeling.ipynb` notebook establishes this baseline approach, while `Modeling2.ipynb` likely focuses on hyperparameter tuning (e.g., `GridSearchCV` or `RandomizedSearchCV`) to optimize each of the four models.

### 3. Prediction & Submission

1.  The test data (`test.csv`) is loaded.
2.  Each of the four trained models (`model_IE`, `model_NS`, etc.) predicts its respective binary class on the test data.
3.  The four binary predictions are mapped back to their letter codes (e.g., `1` -> `'I'`, `0` -> `'E'`).
4.  The final `Personality` string is created by concatenating the four predicted letters.
5.  The results are formatted into `submission.csv` with `id` and `Personality` columns.

## 📈 Hyperparameter Tuning & Results

The models were tuned (likely using Bayesian optimization or a similar method, as seen in `Modeling2.ipynb`) to find the optimal set of parameters.

### Best Parameters Found

```json
{
    "n_estimators": 299,
    "max_depth": 5,
    "learning_rate": 0.15841262137302178,
    "subsample": 0.8519152889164038,
    "colsample_bytree": 0.6808885474211932,
    "gamma": 2.0070959113867732,
    "reg_alpha": 1.2522715414146957,
    "reg_lambda": 2.5895571241033593
}
```

### Performance Metrics
- Best CV Score: 0.9639316464361883
- Tuned Model R2 (Train): 0.9640534903692799
- Tuned Model RMSE: 0.30210824789781193

---

## 💻 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RaymussenArthur/Personality-Prediction.git](https://github.com/RaymussenArthur/Personality-Prediction.git)
    cd Personality-Prediction
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install pandas numpy scikit-learn xgboost jupyter
    ```

3.  **Get the data:**
    * Download the `train.csv`, `test.csv`, and `sample_submission.csv` files from the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s5e7/data).
    * Place them inside the `/data` folder.

4.  **Run the notebooks:**
    * Start Jupyter:
        ```bash
        jupyter notebook
        ```
    * Open and run the notebooks in the `Notebooks/` directory, starting with `Modeling.ipynb` and `Modeling2.ipynb` to train models and generate a submission.
