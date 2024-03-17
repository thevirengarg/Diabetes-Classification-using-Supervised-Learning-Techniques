# Diabetes-Classification-using-Supervised-Learning-Techniques

This repository contains a Jupyter notebook for building machine learning models to predict the presence of diabetes in Pima Indian women based on various diagnostic measurements. The dataset used in this project is the Pima Indians Diabetes Database, which is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

## Dataset

The dataset consists of several medical predictor variables and one target variable, `Outcome`, which indicates whether a patient has diabetes or not. The predictor variables include:

- Number of pregnancies
- Plasma glucose concentration
- Diastolic blood pressure
- Triceps skin fold thickness
- 2-Hour serum insulin
- Body mass index (BMI)
- Diabetes pedigree function
- Age
- Outcome(indicating diabetes or not)

The dataset is available in the `data` directory of this repository as `diabetes.csv`.

## Requirements

To run this project, you'll need the following dependencies:

- Python (version 3.6 or later)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn (for handling imbalanced data)

## Code Analysis

The `Diabetes.ipynb` notebook performs the following tasks:

1. **Data Loading and Preprocessing**:
   - Loads the dataset from the `diabetes.csv` file.
   - Handles missing values by imputing with the minimum value or using iterative imputation with random forest regression.
   - Scales the features using StandardScaler and RobustScaler (although it doesn't affect the model performance).
   - The 'Pregnancies' feature was dropped from the dataset as it was deemed potentially irrelevant to predicting diabetes.
   - Performs exploratory data analysis, including visualizations and statistical summaries.

2. **Model Training and Evaluation**:
   - Splits the data into training and testing sets.
   - Trains a logistic regression model with different solvers and hyperparameters.
   - Trains a support vector machine (SVM) model with different kernels.
   - Evaluates the models' performance using metrics like accuracy, classification report, confusion matrix, and ROC-AUC score.

3. **Visualization**:
   - Generates visualizations, such as scatter plots, histograms, heatmaps, and ROC curves, to assess the models' performance and data distribution.

4. **Handling Imbalanced Data**:
   - Attempts to handle imbalanced data using oversampling with SMOTE (Synthetic Minority Over-sampling Technique), but it doesn't improve the model's performance.


## Conclusion
- Logistic regression and SVM models were trained to predict diabetes based on health metrics.
- Logistic regression with the 'lbfgs' solver performed consistently well across different metrics.
- SVM models showed comparable performance, with the linear kernel performing slightly better.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

