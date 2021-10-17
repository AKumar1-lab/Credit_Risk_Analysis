## Credit_Risk_Analysis

Module 17 Supervised Machine Learning and Credit Risk

Completed by Angela Kumar

### Purpose
Predict credit risk with machine learning models that are built and evaluated using Python

### Overview
Use Python to build and evaluate several machine learning models to predict credit risk. Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.

### Resources

Data: LoanStats_2019Q1.csv; credit_risk_resampling_starter_code.ipynb converted to credit_risk_resampling.ipynb; 
credit_risk_ensemble_starter_code.ipynb converted to credit_risk_ensemble.ipynb

Technologies:  Python; Juypter Notebook MLENV; Anaconda MLENV; imbalanced-learn libraries;scikit-learn libraries; 

### Background

Jill commends you for all your hard work. Piece by piece, you’ve been building up your skills in data preparation, statistical reasoning, and machine learning. You are now ready to apply machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, you’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use **imbalanced-learn** and **scikit-learn** libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, you’ll oversample the data using the **RandomOverSampler** and **SMOTE** algorithms, and undersample the data using the **ClusterCentroids** algorithm. Then, you’ll use a combinatorial approach of over- and undersampling using the **SMOTEENN** algorithm. Next, you’ll compare two new machine learning models that reduce bias, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**, to predict credit risk. Once you’re done, you’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

### Deliverables

* Deliverable 1: Use Resampling Models to Predict Credit Risk
* Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
* Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
* Deliverable 4: A Written Report on the Credit Risk Analysis

### Deliverable 1: Use Resampling Models to Predict Credit Risk

Using your knowledge of the imbalanced-learn and scikit-learn libraries, you’ll evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, you’ll use the oversampling **RandomOverSampler** and **SMOTE** algorithms, and then you’ll use the undersampling **ClusterCentroids** algorithm. Using these algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

* 1. Clean the data and create a dataframe based on the initial credit data

<img width="735" alt="initial loan_status dataframe" src="https://user-images.githubusercontent.com/85860367/137610393-69cc10f2-2721-4ce7-b839-97713510d179.PNG">

* 2. Create the training and target variables

<img width="715" alt="Create variable and target" src="https://user-images.githubusercontent.com/85860367/137610785-89c50141-d9c7-4389-8796-337263b9436f.PNG">

* 3. Check the number of targets

<img width="344" alt="Number of loan status" src="https://user-images.githubusercontent.com/85860367/137610819-f9811672-a644-4563-b9f9-88958b4f8b7c.PNG">

* 4. RandomOverSampler(oversampling):

![image](https://user-images.githubusercontent.com/85860367/137611164-ca1dd04a-4ebf-4449-9282-58b12590694f.png)


SMOTE:

ClusterCentroids: 

### Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

Using your knowledge of the imbalanced-learn and scikit-learn libraries, you’ll use a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the **SMOTEENN** algorithm, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

### Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

Using your knowledge of the imblearn.ensemble library, you’ll train and compare two different ensemble classifiers, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**, to predict credit risk and evaluate each model. Using both algorithms, you’ll resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

### Deliverable 4: Written Report on the Credit Risk Analysis

For this deliverable, you’ll write a brief summary and analysis of the performance of all the machine learning models used in this Challenge.

### Summary

### Recommendation
