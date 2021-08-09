# About the Project
Through the series of experiments, our goal was to present a model with the best
prediction. There were many different algorithms to choose from starting from
linear regression, subset selection, regularization to decision trees, and
Random forests. But as the “No Free Lunch Policy” goes, each algorithm has its
strengths and weaknesses, and we can use these properties to choose an algorithm
and tune it to meet our goal.

**The Objective**:

Our goal is to obtain the best **predictions** from unseen data using different
statistical learning algorithms.

**The Problem**:

The training data set for this project consists of 550 data points as rows and 8
columns as features. The test data consists of 218 data points and 8 columns as
features. To understand the concepts of data analytics and explore various
methodologies behind it, the team was tasked to build several models to predict
the response for the data and select the best 3 amongst them.

**Key Highlights**:

We have researched extensively to learn about different statistical learning
methods. Initially, we tried fitting simple linear models and simple decision
trees. Because of non-linearity in data, we tried using models that can handle
non-linearity better. We used different variations of decision trees and
concluded that boosting is the best method that is aligned to our objectives.
Tuning of hyper-parameters and Cross-Validation played an important role in the
process. In our project, data analysis is done with R Statistical Software.

**The Solution Procedure**:

-   Visualizing data and looking for correlations and non-linearities

-   Recognition of suitable models

-   Optimization of models

-   Evaluation of optimized models using the same metric (test error on
    validation set)

-   Choosing the best model

**Results**:

The 3 best models obtained with the lowest training error rates were MLR,
Bagging and Boosting. On validation data, MLR, Bagging and Boosting resulted in
a test error **of 2.087, 0.346** and **0.125** respectively. Boosting was chosen
as the best model owing to its lowest error rate. The test data resulted in an
MSE of **0.266** using the boosting model. (Caret package gives different
results with different platforms. We are yet to figure out a way to manage that
issue)

-   Best model = **Tree Boosting**

-   Best test error for unseen data = **0.266 (Model with Improvement)**

-   Best test error on validation set = **0.125**

Please go to [report](https://github.com/ridhampatoliya/Engineering-Data-Analysis---Class-Project/blob/master/Final_Report.pdf) for more information :)

