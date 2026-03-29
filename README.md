# Engineering Data Analysis Class Project

A graduate-level statistical learning project completed for **ISEN 613 (Spring 2021)**. This work compares multiple predictive modeling approaches in **R** and selects the strongest model for forecasting outcomes on unseen engineering data.

The project moves from baseline linear methods to more flexible tree-based ensembles, with a focus on model selection, validation performance, and practical predictive accuracy. Among the models tested, **Tree Boosting** produced the best overall results.

## Key Takeaways

- Built and compared several statistical learning models on the same engineering dataset
- Evaluated models using a consistent validation framework
- Identified **Boosting** as the strongest-performing approach
- Achieved a **validation error of 0.125** and **test MSE of 0.266** with the final model
- Documented both the comparative study and the final selected model in separate markdown reports

## Project Overview

The goal of this project was to develop predictive models using a labeled training dataset, compare their performance, and select the best model for generalization on unseen data.

The analysis explored a range of methods, including:

- Multiple Linear Regression
- Subset Selection / Regularization
- Decision Trees
- Bagging
- Boosting

This repository captures both the model comparison process and the final boosted-tree solution selected from that process.

## Problem Statement

The dataset provided for this project includes:

- **Training set:** 550 observations and 8 input features
- **Test set:** 218 observations and 8 input features

The objective was to train multiple statistical learning models, tune them appropriately, compare their predictive performance, and choose the strongest candidates for final evaluation.

## Methodology

The project followed a structured model development workflow:

1. Explore the dataset to identify correlations and nonlinear patterns
2. Build candidate models using multiple statistical learning methods
3. Tune model hyperparameters and improve fit
4. Evaluate each model using a common validation metric
5. Select the best-performing model for final prediction on unseen data

The full analysis was conducted in **R Statistical Software**.

## Results Summary

| Model | Validation Error |
| --- | ---: |
| Multiple Linear Regression | 2.087 |
| Bagging | 0.346 |
| Boosting | 0.125 |

### Final Selected Model

- **Best model:** Tree Boosting
- **Best validation error:** 0.125
- **Test MSE on unseen data:** 0.266

Boosting outperformed the other candidate models and was selected as the final approach because it handled nonlinear structure in the data more effectively than the simpler baseline methods.

## Repository Structure

- `01_Model_Selection_and_comparative_study.md`  
  Comparative analysis of candidate models and the model selection process
- `02_Final_Model_Boosted_Tree.md`  
  Detailed write-up of the final boosted tree model
- `Final_Report.pdf`  
  Full report containing methodology, experiments, and conclusions
- `train.xlsx`  
  Training dataset
- `test.xlsx`  
  Test dataset
- `r_project_metadata`  
  R project metadata and supporting project files

## Tools Used

- **Language:** R
- **Focus areas:** Statistical learning, predictive modeling, model comparison
- **Techniques:** Cross-validation, hyperparameter tuning, nonlinear modeling, ensemble learning

## Why This Project Matters

This project demonstrates practical experience with:

- Comparing multiple machine learning models instead of relying on a single approach
- Selecting models based on validation performance rather than intuition alone
- Working with nonlinear data and ensemble methods
- Presenting analytical findings in both technical and report-ready formats

## Notes

Results may vary slightly across environments due to platform-specific behavior in certain R package implementations.

## Author

**Ridham Patoliya**

## Course Context

Completed as a class project for **ISEN 613** during the **Spring 2021** semester.
