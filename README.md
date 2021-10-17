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

##### Clean the data and create a dataframe based on the initial credit data

<img width="420" alt="initial loan_status dataframe" src="https://user-images.githubusercontent.com/85860367/137610393-69cc10f2-2721-4ce7-b839-97713510d179.PNG">

##### Create the training and target variables

<img width="420" alt="Create variable and target" src="https://user-images.githubusercontent.com/85860367/137610785-89c50141-d9c7-4389-8796-337263b9436f.PNG">

##### Check the number of targets

<img width="344" alt="Number of loan status" src="https://user-images.githubusercontent.com/85860367/137610819-f9811672-a644-4563-b9f9-88958b4f8b7c.PNG">

##### 1. RandomOverSampler(oversampling):

<img width="420" alt="RandomOverSampling" src="https://user-images.githubusercontent.com/85860367/137611868-7c973355-5aa7-435b-ac94-f225845c7975.PNG">

##### 2. Synthetic Minority Oversampling Technique (SMOTE)(oversampling):
                                                                                                                                                     
<img width="420" alt="SMOTEOversampling" src="https://user-images.githubusercontent.com/85860367/137611695-d87e6f1c-8107-4a06-b439-1b39aeba8e58.PNG">

##### 3. ClusterCentroids (undersampling): 

<img width="420" alt="ClusterCentroids screenshot" src="https://user-images.githubusercontent.com/85860367/137646863-b5623407-57f8-41c7-a2dd-f5c01c51ecbc.PNG">

### Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

Using your knowledge of the imbalanced-learn and scikit-learn libraries, you’ll use a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the **SMOTEENN** algorithm, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

##### 4. SMOTEENN Sampling (Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.)

<img width="420" alt="SMOTEENNSampling" src="https://user-images.githubusercontent.com/85860367/137612073-245b5f79-d16b-4eee-80cb-9b9b39b7eced.PNG">

### Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

Using your knowledge of the imblearn.ensemble library, you’ll train and compare two different ensemble classifiers, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**, to predict credit risk and evaluate each model. Using both algorithms, you’ll resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

##### 5. BalancedRandomForestClassifer(A balanced random forest randomly under-samples each boostrap sample to balance it to reduce bias)

<img width="420" alt="Balanced Random Forest Classifer" src="https://user-images.githubusercontent.com/85860367/137612581-d514380f-4f0c-4496-ba64-eb8f44f19443.PNG">

##### 6. EasyEnsembleClassifer(The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling)

<img width="420" alt="EasyEnsemble AdaBoosterClassiferSample" src="https://user-images.githubusercontent.com/85860367/137612589-42b4a2cc-0cab-4c98-9e60-1c6c4900c666.PNG">

### Deliverable 4: Written Report on the Credit Risk Analysis

For this deliverable, you’ll write a brief summary and analysis of the performance of all the machine learning models used in this Challenge.

### Summary

![image](https://user-images.githubusercontent.com/85860367/137616264-af3eaf4d-ed0f-488d-af76-661cd20d5e13.png)

Accuracy is defined as the percentage of correct predictions (True Positive = True Negative/True Positive+ True Negative+ False Positive + False Negative)

Precision is defined as the perentage of positive predictions that are correct(True Positive/(True Positive + False Positive)).  It is the measure of how reliable the positive classification is.

Sensitivity(Recall) is defined as the percentage of true positive predictions that are correct (True Positive/(True Positive + False Negative)).  It is the measure of the number of observations with a positive classification will be correctly diagnosed.

F1 Score is defined as the harmonic mean that balances the precision and sensitivity F1 = 2(precision * sensitivity)/ (precision + sensitivity)

The analysis show that the precision for the high-risk loans were low for all of the models. The sensitivity(recall) for the high-risk loans was different among all of the models.  In some cases it was either the model was overfitted or underfitted.  

The ClusterCentroid was an example where the high-risk loans underfitted, there is a huge spread in the sensitivity between high and low-risk loans and will have poor results as it does have the correct results or new data cannot be analyed by the model as features are not selected appropriately and only 54% was the balanced accuracy.  

Naive Random OverSampling and SMOTEENN models was an example where the high-risk loans were overfitted, it is a possibility that some results or new data may not be picked up in the model. The machine memorized the algorithms, it appears there were more true negatives in the array, and only 66%, and 64% balanced accuracy, respectively.

### Recommendation
Easy Ensemble Ada Boost Classifier model had the best accuracy of 93.17% which means the algorithm was just right as well as the sensitivity between high and low-risk loans.  The spread was minimal, the F1 Score was also the highest.  The bias in this model was also reduced, so all data existing and new was analyzed, which had a better result or outcome.
