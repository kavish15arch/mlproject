
## First ML project
## Overview
This machine learning project predicts the likelihood of a student's admission into a university based on their academic credentials and exam scores. The goal is to provide a data-driven estimation tool that helps prospective students gauge their chances of acceptance.

## Dataset
* **Source:** Students Performance in Exams (Kaggle)
* **Size:** 1,000 records, 8 features
* **Input Features (Predictors):** 
  * `gender`, `race_ethnicity`, `parental_level_of_education`
  * `lunch` (Standard vs. Free/Reduced)
  * `test_preparation_course` (None vs. Completed)
* **Target Variables:** `math_score`, `reading_score`, `writing_score`

## Methodology
* **Data Preprocessing:** Handled missing values, standardized numerical features, and applied One-Hot Encoding for categorical demographic variables (gender, race/ethnicity, lunch type, etc.).
* **Models Tested:** Evaluated a comprehensive suite of regression algorithms to predict student scores:
  * Linear Regression, Ridge Regression, and Lasso Regression
  * Random Forest Regressor
  * K-Neighbors Regressor
  * AdaBoost Regressor
  * Gradient Boosting Regressor
* **Hyperparameter Tuning:** Optimized model performance by defining parameter grids and tuning key hyperparameters (e.g., `n_estimators`, `learning_rate`, and `weights` for tree-based and distance-based models).
* **Final Selection:** Selected [Winning Model] as the final model because it achieved the best balance of variance and bias, resulting in the highest R-squared score and lowest Mean Absolute Error on the test data.

## Results
* **Accuracy:** [e.g., 88%]
* **Key Insight:** [e.g., CGPA and GRE scores were found to have the highest feature importance in predicting successful admission.]

## Project Structure
* `src/pipeline/`: Contains the modular code for data ingestion, training, and prediction.
* `app.py`: The main script to launch the user interface.
* `requirements.txt`: Dependencies required to replicate the environment.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
