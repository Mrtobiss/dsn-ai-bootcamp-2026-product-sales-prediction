# Product Sales Prediction — DSN AI Bootcamp 2026

**DSN AI Bootcamp Qualification Hackathon 2026 — ML Track**

**Author:** Ibrahim Yisau

## Project Overview

This project develops a machine learning regression model to predict `total_sales` for product–store combinations in the **DSN AI Bootcamp 2026 Product Sales Prediction Challenge**.

The dataset contains product characteristics, pricing information, and store attributes. The objective is to use the available training data to predict sales for product–store combinations in the test set.

The project follows an end-to-end machine learning workflow:

> **Understand → Analyse → Prepare → Model → Validate → Tune → Predict → Submit**

## Problem Statement

The task is a **supervised regression problem**.

Each observation represents a product sold through a particular store, with `total_sales` as the target variable.

The training dataset contains:

* **6,818 observations**
* **12 predictor variables**
* **1 target variable (`total_sales`)**

The test dataset contains:

* **1,705 observations**
* The same predictor variables without the target

The required submission contains:

* `id`
* `total_sales`

## Dataset Features

The predictors include:

| Feature               | Description                             |
| --------------------- | --------------------------------------- |
| `product_code`        | Product identifier                      |
| `product_weight_kg`   | Product weight                          |
| `fat_content`         | Product fat-content category            |
| `shelf_visibility`    | Product visibility on the store shelf   |
| `product_category`    | Product category                        |
| `product_price`       | Product price                           |
| `store_code`          | Store identifier                        |
| `store_age_years`     | Store age                               |
| `store_size`          | Store size category                     |
| `store_location_tier` | Store location tier                     |
| `store_format`        | Store format                            |
| `id`                  | Row identifier, excluded from modelling |

The target variable is:

`total_sales`

## Exploratory Data Analysis

The analysis examined:

* Missing values
* Duplicate observations
* Categorical distributions
* Target distribution
* Numerical feature distributions
* Correlation between numerical variables and sales
* Sales differences across categorical variables
* Store-level sales patterns
* Product-level consistency
* Product–store combinations
* Train/test overlap

One notable finding was that `product_category` contained inconsistent capitalization and whitespace. Standardizing the values reduced the number of raw categories and provided a cleaner categorical representation.

The analysis also showed that the test set contains previously observed products and stores, but the **product–store combinations themselves were not present in the training data**.

## Data Preparation

The preprocessing pipeline included:

* Preserving the original datasets
* Standardizing `product_category`
* Handling missing categorical values using an explicit `"Missing"` category
* Keeping numerical values unscaled because tree-based models do not require feature scaling
* Maintaining consistent train/test feature columns
* Excluding `id` from model training because it is a row identifier rather than a predictive feature

## Modelling Approach

The primary model used was **CatBoost Regressor**.

CatBoost was selected because the dataset contains several categorical variables and CatBoost can handle categorical features directly.

Several experiments were conducted before selecting the final configuration.

### Feature Engineering Experiments

The following approaches were tested:

* Missing `product_weight_kg` indicator
* Product-level aggregate features
* Numerical interaction features
* Categorical interaction features
* Log-transformed target

None of these experiments improved the depth-8 CatBoost configuration on the initial 80/20 validation split, so they were not included in the final model.

## Model Validation

The modelling process progressed from a single 80/20 holdout split to **5-fold cross-validation**.

The target was divided into approximately equal-frequency bins to allow stratified folds for the continuous regression target.

### Model Comparison

The following models were evaluated using the same cross-validation folds and feature representation:

| Model    | Mean CV RMSE | Std CV RMSE |
| -------- | -----------: | ----------: |
| CatBoost |    1074.0863 |     18.9655 |
| LightGBM |    1091.1101 |     18.4788 |
| XGBoost  |    1093.5405 |     16.1063 |

Within the configurations tested in this project, CatBoost produced the lowest mean cross-validation RMSE.

## Hyperparameter Experiments

CatBoost hyperparameters were tested independently for:

* Tree depth
* Learning rate
* L2 leaf regularization

A combined candidate configuration was subsequently evaluated directly using the same 5-fold cross-validation setup.

The benchmark configuration remained the selected configuration after this comparison.

## Final Model

The final model was trained using:

```text
Model: CatBoostRegressor

Depth: 8
Learning rate: 0.05
L2 leaf regularization: 3
Iterations: 126
Random seed: 42
Loss function: RMSE
```

The final model was retrained on the **complete training dataset** before generating predictions for the test set.

### Cross-Validation Performance

The retained CatBoost configuration achieved:

**Mean CV RMSE: 1074.0863**

**CV standard deviation: 18.9655**

These figures represent performance on the project's 5-fold validation procedure and should not be interpreted as the competition leaderboard score.

## Submission

The final model generated predictions for all:

**1,705 test observations**

The resulting submission file contains:

```text
id,total_sales
```

The generated file was saved as:

```text
submission.csv
```

The submission was checked for:

* Correct number of rows
* Required columns
* Missing predictions
* Infinite predictions
* Correct `id` alignment

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* LightGBM
* XGBoost
* Matplotlib
* Google Colab
* GitHub

## Key Learning Outcomes

This project provided practical experience with:

* Regression modelling
* Exploratory data analysis
* Categorical feature handling
* Missing-data handling
* Feature engineering
* Cross-validation for regression
* Model comparison
* Hyperparameter experimentation
* Early stopping
* Training a final model on the complete dataset
* Generating and validating competition submissions

## Reproducibility

The notebook is designed to run sequentially from top to bottom.

The expected input files are:

```text
train.csv
test.csv
sample_submission.csv
```

After placing these files in the working directory, the notebook can be executed to reproduce the modelling workflow and generate:

```text
submission.csv
```

## Author

**Ibrahim Yisau**

AI / Machine Learning | Data Science | Research & Technology for Development

GitHub: **Mrtobiss**
