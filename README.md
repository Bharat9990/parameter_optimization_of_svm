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

*****************************************************************************************
## grid.cv_results_ = 
{'mean_fit_time': array([0.10288434, 0.77454166, 0.5540648 , 0.11442785, 1.13957572,
        0.11483302, 0.10675616, 0.75999174, 0.53661785, 0.10300903,
        1.12609234, 0.10985813, 0.10228419, 0.77532177, 0.59243822,
        0.10961595, 1.17148542, 0.11401539]),
 'std_fit_time': array([0.00612162, 0.02364219, 0.02832665, 0.01650438, 0.03089791,
        0.0106876 , 0.00736087, 0.01616052, 0.01880507, 0.00981854,
        0.0090336 , 0.01070702, 0.01608321, 0.0179488 , 0.05510725,
        0.02593254, 0.0363066 , 0.01473998]),
 'mean_score_time': array([0.01447783, 0.78653631, 0.12706943, 0.01715984, 0.90264449,
        0.01626401, 0.01157413, 0.75858307, 0.11330667, 0.01069407,
        0.88820128, 0.01582117, 0.01485467, 0.76014442, 0.11687899,
        0.01298866, 0.89735918, 0.01720743]),
 'std_score_time': array([0.00323424, 0.01589579, 0.02055605, 0.00276124, 0.03574363,
        0.00205578, 0.00273815, 0.02246158, 0.02140248, 0.00343798,
        0.03153072, 0.0021264 , 0.00361255, 0.02063466, 0.01423009,
        0.00120074, 0.02070661, 0.00270919]),
 'param_C': masked_array(data=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 10, 10,
                    10, 10, 10, 10],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False],
        fill_value='?',
             dtype=object),
 'param_gamma': masked_array(data=['scale', 'scale', 'scale', 'auto', 'auto', 'auto',
                    'scale', 'scale', 'scale', 'auto', 'auto', 'auto',
                    'scale', 'scale', 'scale', 'auto', 'auto', 'auto'],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False],
        fill_value='?',
             dtype=object),
 'param_kernel': masked_array(data=['linear', 'rbf', 'poly', 'linear', 'rbf', 'poly',
                    'linear', 'rbf', 'poly', 'linear', 'rbf', 'poly',
                    'linear', 'rbf', 'poly', 'linear', 'rbf', 'poly'],
              mask=[False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False],
        fill_value='?',
             dtype=object),
 'params': [{'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
  {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'},
  {'C': 0.1, 'gamma': 'scale', 'kernel': 'poly'},
  {'C': 0.1, 'gamma': 'auto', 'kernel': 'linear'},
  {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf'},
  {'C': 0.1, 'gamma': 'auto', 'kernel': 'poly'},
  {'C': 1, 'gamma': 'scale', 'kernel': 'linear'},
  {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
  {'C': 1, 'gamma': 'scale', 'kernel': 'poly'},
  {'C': 1, 'gamma': 'auto', 'kernel': 'linear'},
  {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'},
  {'C': 1, 'gamma': 'auto', 'kernel': 'poly'},
  {'C': 10, 'gamma': 'scale', 'kernel': 'linear'},
  {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
  {'C': 10, 'gamma': 'scale', 'kernel': 'poly'},
  {'C': 10, 'gamma': 'auto', 'kernel': 'linear'},
  {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
  {'C': 10, 'gamma': 'auto', 'kernel': 'poly'}],
 'split0_test_score': array([0.54671115, 0.45328885, 0.45328885, 0.54671115, 0.9313632 ,
        0.45328885, 0.45328885, 0.45328885, 0.45328885, 0.45328885,
        0.93231649, 0.45328885, 0.45328885, 0.45328885, 0.45328885,
        0.45328885, 0.66968541, 0.45328885]),
 'split1_test_score': array([0.44778255, 0.45493562, 0.45302814, 0.44778255, 0.68144969,
        0.45255126, 0.52932761, 0.51072961, 0.45255126, 0.52932761,
        0.94420601, 0.45255126, 0.54744874, 0.51072961, 0.45255126,
        0.54744874, 0.6948021 , 0.45255126]),
 'split2_test_score': array([0.56604673, 0.45398188, 0.45350501, 0.56604673, 0.93037673,
        0.50309967, 0.45302814, 0.45398188, 0.45398188, 0.45302814,
        0.9375298 , 0.50309967, 0.53266571, 0.45398188, 0.48497854,
        0.53266571, 0.64806867, 0.50309967]),
 'split3_test_score': array([0.45302814, 0.45302814, 0.45302814, 0.45302814, 0.93180734,
        0.45302814, 0.55793991, 0.45302814, 0.45302814, 0.55793991,
        0.93228422, 0.45302814, 0.45302814, 0.45302814, 0.45302814,
        0.45302814, 0.58798283, 0.45302814]),
 'split4_test_score': array([0.45302814, 0.45302814, 0.45302814, 0.45302814, 0.58750596,
        0.45302814, 0.54697186, 0.45302814, 0.45302814, 0.54697186,
        0.92799237, 0.45302814, 0.54697186, 0.45302814, 0.45302814,
        0.54697186, 0.60133524, 0.45302814]),
 'mean_test_score': array([0.49331934, 0.45365252, 0.45317565, 0.49331934, 0.81250059,
        0.46299921, 0.50811127, 0.46481132, 0.45317565, 0.50811127,
        0.93486578, 0.46299921, 0.50668066, 0.46481132, 0.45937498,
        0.50668066, 0.64037485, 0.46299921]),
 'std_test_score': array([0.05188511, 0.00073031, 0.00019317, 0.05188511, 0.14836046,
        0.02005164, 0.04578826, 0.0229618 , 0.00046823, 0.04578826,
        0.00556316, 0.02005164, 0.04402254, 0.0229618 , 0.01280399,
        0.04402254, 0.04037238, 0.02005164]),
 'rank_test_score': array([ 8, 16, 17,  8,  2, 12,  4, 10, 17,  4,  1, 12,  6, 10, 15,  6,  3,
        12]),
 'split0_train_score': array([0.54697186, 0.45314735, 0.45326657, 0.54697186, 0.97711016,
        0.45302814, 0.45302814, 0.45314735, 0.45326657, 0.45302814,
        0.97782546, 0.45302814, 0.45302814, 0.45314735, 0.45326657,
        0.45302814, 0.7739628 , 0.45302814]),
 'split1_train_score': array([0.44677554, 0.4519013 , 0.45333174, 0.44677554, 0.7724401 ,
        0.45333174, 0.53248301, 0.49791393, 0.45333174, 0.53248301,
        0.97699368, 0.45333174, 0.54666826, 0.49791393, 0.45333174,
        0.54666826, 0.80271784, 0.45333174]),
 'split2_train_score': array([0.56299917, 0.45297413, 0.45297413, 0.56299917, 0.97699368,
        0.52306592, 0.45297413, 0.45297413, 0.45297413, 0.45297413,
        0.97806652, 0.52306592, 0.53820479, 0.45297413, 0.48301347,
        0.53820479, 0.74919537, 0.52306592]),
 'split3_train_score': array([0.45321254, 0.45321254, 0.45333174, 0.45321254, 0.97496722,
        0.45333174, 0.5448802 , 0.45321254, 0.45333174, 0.5448802 ,
        0.97639766, 0.45333174, 0.45333174, 0.45321254, 0.45333174,
        0.45333174, 0.68613661, 0.45333174]),
 'split4_train_score': array([0.45321254, 0.45321254, 0.45333174, 0.45321254, 0.69841459,
        0.45321254, 0.54690666, 0.45321254, 0.45333174, 0.54690666,
        0.97806652, 0.45321254, 0.54690666, 0.45321254, 0.45333174,
        0.54690666, 0.69614972, 0.45321254]),
 'mean_train_score': array([0.49263433, 0.45288957, 0.45324719, 0.49263433, 0.87998515,
        0.46719402, 0.50605443, 0.4620921 , 0.45324719, 0.50605443,
        0.97746997, 0.46719402, 0.50762792, 0.4620921 , 0.45925505,
        0.50762792, 0.74163247, 0.46719402]),
 'std_train_score': array([0.05121516, 0.00050178, 0.00013884, 0.05121516, 0.12033231,
        0.02793617, 0.04359844, 0.01791113, 0.00013884, 0.04359844,
        0.00066555, 0.02793617, 0.04456708, 0.01791113, 0.01187923,
        0.04456708, 0.04468186, 0.02793617])}
