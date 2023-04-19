# parameter_optimization_of_svm
## Parameter Optimization of Support Vector Machines
Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.

Some of the most important parameters of SVM such as kernel, C, and gamma can be changed in order to achieve a higher accuracy. This is called as Hyperparameter Tuning.

We can perform this task using GridSearchCV for optimizing these parameters.

In this python file, I've used a Fitness Function to optimize the parameters.

## DATASET -> EEG Eye State Dataset - 14980 rows * 15 columns

All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analysing the video frames. '1' indicates the eye-closed and '0' the eye-open state. All values are in chronological order with the first measured value at the top of the data.

## Best model parameters - 
### Sample 1 - Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
### Sample 2 - Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
### Sample 3 - Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
### Sample 4 - Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
### Sample 5 - Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
### Sample 6 - Best Parameters: {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf'}
### Sample 7 - Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
### Sample 8 - Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
### Sample 9 - Best Parameters: {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf'} 
### Sample 10 - Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'} 
## Train Accuracy - Mean: 0.96, SD: 0.03
## Test Accuracy - Mean: 0.92, SD: 0.04
![image](https://user-images.githubusercontent.com/110338556/233210814-52950d35-5a0e-4715-9953-ca0d9182f73c.png)

*****************************************************************************************
