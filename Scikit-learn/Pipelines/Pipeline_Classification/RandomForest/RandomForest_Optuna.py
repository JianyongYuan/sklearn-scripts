#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import optuna
import pandas as pd
import numpy as np
import datetime,time,os,sys,joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn_pandas import DataFrameMapper


###############  Some user-defined functions  ###############
def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time,2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))

def model_choice(model_name, model_params):
    global ML_type
    if model_name == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**model_params)
        ML_type = "Regression"
        return "rfr", model
    elif model_name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**model_params)
        ML_type = "Classification"
        return "rfc", model
    else:
        print('** Please choose a RrandomForest model **\n-> Supported RandomForest models are as follows:')
        print('   (1) RandomForestRegressor\n   (2) RandomForestClassifier')
        exit(1)

def show_metrics(model, ML_type, y_train_pred, y_train, y_test_pred, y_test_pred_proba, y_test, X_train, X_test):
    print("\n            >>>>  Metrics based on the best model  <<<<\n")
    if ML_type == "Classification":
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        print('> Accuracy on the training set:  {:.2%}'.format(accuracy_train))
        print('> Accuracy on the test set:  {:.2%}'.format(accuracy_test))
        print('> Score on the training set:  {:.2%}'.format(model.score(X_train, y_train)))
        print('> Score on the test set:  {:.2%}'.format(model.score(X_test, y_test)))
        print('> Classification report on the training set:')
        print(classification_report(y_train, y_train_pred))
        print('> Classification report on the test set:')
        print(classification_report(y_test, y_test_pred))

        roc_auc_test, average_precision_test = [], []
        for i in range(len(set(y_train))):
            roc_auc_test.append(roc_auc_score(y_test, y_test_pred_proba[:,i], multi_class='ovr'))
            average_precision_test.append(average_precision_score(y_test, y_test_pred_proba[:,i]))
        pd.set_option('display.float_format','{:12.6f}'.format)
        pd.set_option('display.colheader_justify', 'center')
        test_reports = pd.DataFrame(np.vstack((roc_auc_test, average_precision_test)).T, columns=['ROC-AUC','AP(PR-AUC)'])
        print('> Area under the receiver operating characteristic curve (ROC-AUC) and\n  average precision (AP) which summarizes a precision-recall curve as the weighted mean\n  of precisions achieved at each threshold on the test set:\n  {}\n'.format(test_reports))

    elif ML_type == "Regression":
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        print('> Mean squared error (MSE) on the training set:  {:.6f}'.format(mse_train))
        print('> Mean squared error (MSE) on the test set:  {:.6f}'.format(mse_test))
        print('> Mean absolute error (MAE) on the training set:  {:.6f}'.format(mae_train))
        print('> Mean absolute error (MAE) on the test set:  {:.6f}'.format(mae_test))
        print('> R-squared (R^2) value on the training set:  {:.6f}'.format(model.score(X_train, y_train)))
        print('> R-squared (R^2) value on the test set:  {:.6f}\n'.format(model.score(X_test, y_test)))

def set_sampler(sample_method, seed_number):
    if sample_method == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=seed_number)
    elif sample_method == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=seed_number)
    
    return sampler
    
    
###############  Set required parameters and load data here  ###############
'''basic parameters'''
save_model = True   # the best model is saved as the *_best.pkl file
model_name = "RandomForestClassifier"  # "RandomForestRegressor" or "RandomForestClassifier"
model_params = {'random_state':0, 'class_weight':'balanced'}
scoring_metrics = "accuracy" #   https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
score_opt_direction = "maximize"   # "minimize" or "maximize"
sample_method = "TPESampler"  # "TPESampler" or "RandomSampler"
timeout = None # time limit in seconds for the search of appropriate models. 
n_trials = 3
cv_fold = 3
n_jobs = 8
seed_number = 0  # set a seed number

'''load the dataset'''
selected_features = ['MolWt','NumRotatableBonds','AromaticProportion']
df = pd.read_csv('../../MolLogP_dataset.csv')
data_X = df[selected_features]
data_y = df['MolLogP<2']

# print(data_y)
# exit()

'''preprocessing of the dataset'''
# columns_list_oe = ['ZnO1','ZnO2','ZnO3','DC01','DC01T','WS','PCZ70','SL7025']

# ZnO1 = np.sort(df['ZnO1'].unique()).tolist()
# ZnO2 = np.sort(df['ZnO2'].unique()).tolist()
# ZnO3 = np.sort(df['ZnO3'].unique()).tolist()
# DC01_amount = np.sort(df['DC01'].unique()).tolist()
# DC01T_amount = np.sort(df['DC01T'].unique()).tolist()
# WS_amount = np.sort(df['WS'].unique()).tolist()
# PCZ70_amount = np.sort(df['PCZ70'].unique()).tolist()
# SL7025_amount = np.sort(df['SL7025'].unique()).tolist()

# column_trans = DataFrameMapper(
#                      [
#                       (columns_list_oe, OrdinalEncoder(categories=[ZnO1,ZnO2,ZnO3,DC01_amount,DC01T_amount,WS_amount,PCZ70_amount,SL7025_amount]))],
#                         None
#                         )
# print(column_trans.fit_transform(data_X)[0:20,:])


'''pipeline scheme'''
# pipe_model = Pipeline([('ct', column_trans),
#                     model_choice(model_name, model_params)
#                   ])

pipe_model = Pipeline([
                        model_choice(model_name, model_params)
                      ])


'''hyperparameters to be optimized'''
pipe_params = {}  
pipe_params['rfc__n_estimators'] = optuna.distributions.IntUniformDistribution(30,300,2)
pipe_params['rfc__max_features'] = optuna.distributions.CategoricalDistribution([0.3,0.5,0.8,1.0])
pipe_params['rfc__max_depth'] = optuna.distributions.IntUniformDistribution(5,25,1)
pipe_params['rfc__min_samples_split'] = optuna.distributions.IntUniformDistribution(2,20,1)
pipe_params['rfc__min_samples_leaf'] = optuna.distributions.IntUniformDistribution(1,20,1)
pipe_params['rfc__bootstrap'] = optuna.distributions.CategoricalDistribution([True, False])
#pipe_params['rfc__ccp_alpha'] = optuna.distributions.UniformDistribution(0,10)


###############  The ML training script starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
print('***  Scikit-learn script ({0}) started at {1}  ***\n'.format(model_name, start_date.strftime("%Y-%m-%d %H:%M:%S")))

'''split training/test sets'''
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

study = optuna.create_study(sampler=set_sampler(sample_method, seed_number), direction=score_opt_direction, study_name=model_name)
optuna_search = optuna.integration.OptunaSearchCV(pipe_model, pipe_params, cv=cv_fold, scoring=scoring_metrics, n_jobs=n_jobs, n_trials=n_trials, study=study, timeout=timeout)

optuna_search.fit(X_train, y_train)

best_trial = optuna_search.best_trial_
n_trials = optuna_search.n_trials_
best_score = optuna_search.best_score_
best_params = optuna_search.best_params_
print("\n-> Best trial information: {}\n".format(best_trial))
print("-> Sampling algorithm: {}".format(study.sampler.__class__.__name__))
print("-> Number of completed trials: {}".format(n_trials)) 
print("-> Best CV score: {:.5f}  ({})".format(best_score, scoring_metrics)) 
print("-> Best params: ")
for key, value in best_params.items(): 
    print("  {:>25s}: {}".format(key, value))

best_estimator = optuna_search.best_estimator_
y_train_pred = best_estimator.predict(X_train)
y_test_pred = best_estimator.predict(X_test)
y_test_pred_proba = best_estimator.predict_proba(X_test) if ML_type == "Classification" else None
show_metrics(best_estimator, ML_type, y_train_pred, y_train, y_test_pred, y_test_pred_proba, y_test, X_train, X_test)

if save_model == True:
    filename = sys.argv[0].split(os.sep)[-1].split(".")[0]
    file_pkl = filename + "_best.pkl"
    joblib.dump(best_estimator, file_pkl)
    print('~~ Target model is saved as the \'{0}\' file in the current working directory (CWD) ~~'.format(file_pkl))
    print('   CWD: {0}\n'.format(os.getcwd()))

end_time = time.time()
end_date = datetime.datetime.now()
print('***  Scikit-learn script ({0}) terminated at {1}  ***\n'.format(model_name, end_date.strftime("%Y-%m-%d %H:%M:%S")))
total_running_time(end_time, start_time)


