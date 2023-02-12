#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import pandas as pd
import numpy as np
import datetime,time,joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn_pandas import DataFrameMapper


###############  Set required parameters and load data here  ###############
'''basic parameters'''
filename_pkl = 'HistGradientBoosting_Optuna_best'  # load target model from the *.pkl file
split_dataset = False  # whether to split the dataset into training and test sets
model_name = "HistGradientBoostingClassifier"
#  Supported models are as follows:
#    (1) AdaBoostRegressor / AdaBoostClassifier
#    (2) XGBRegressor / XGBClassifier
#    (3) GradientBoostingRegressor / GradientBoostingClassifier
#    (4) HistGradientBoostingRegressor / HistGradientBoostingClassifier
#    (5) RandomForestRegressor / RandomForestClassifier
#    (6) SVR / SVC
#    (7) MLPRegressor / MLPClassifier
#    (8) ElasticNet / LogisticRegression

'''load the dataset'''
selected_features = ['MolWt','NumRotatableBonds','AromaticProportion']
df = pd.read_csv('../../MolLogP_dataset.csv')
data_X = df[selected_features]
data_y = df['MolLogP<2']


# print(data_y)
# exit()


###############  Some user-defined functions  ###############
def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time,2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))

def load_model(model_name, filename_pkl):
    ML_regression_list = ["XGBRegressor", "AdaBoostRegressor", "GradientBoostingRegressor", 
                          "HistGradientBoostingRegressor", "MLPRegressor", 
                          "RandomForestRegressor", "SVR", "ElasticNet"]
    ML_classification_list = ["XGBClassifier", "AdaBoostClassifier", "GradientBoostingClassifier",
                              "HistGradientBoostingClassifier", "MLPClassifier", 
                              "RandomForestClassifier", "SVC", "LogisticRegression"
                              ]

    if model_name in ML_regression_list:
        ML_type = "Regression"  
    elif model_name in ML_classification_list:
        ML_type = "Classification"

    if model_name == "XGBRegressor":
        from xgboost import XGBRegressor
    elif model_name == "XGBClassifier":
        from xgboost import XGBClassifier
    elif model_name == "AdaBoostRegressor":
        from sklearn.ensemble import AdaBoostRegressor
    elif model_name == "AdaBoostClassifier":
        from sklearn.ensemble import AdaBoostClassifier
    elif model_name == "GradientBoostingRegressor":
        from sklearn.ensemble import GradientBoostingRegressor
    elif model_name == "GradientBoostingClassifier":
        from sklearn.ensemble import GradientBoostingClassifier
    elif model_name == "HistGradientBoostingRegressor":
        from sklearn.ensemble import HistGradientBoostingRegressor
    elif model_name == "HistGradientBoostingClassifier":
        from sklearn.ensemble import HistGradientBoostingClassifier
    elif model_name == "MLPRegressor":
        from sklearn.neural_network import MLPRegressor
    elif model_name == "MLPClassifier":
        from sklearn.neural_network import MLPClassifier
    elif model_name == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
    elif model_name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
    elif model_name == "SVR":
        from sklearn.svm import SVR
    elif model_name == "SVC":
        from sklearn.svm import SVC
    elif model_name == "ElasticNet":
        from sklearn.linear_model import ElasticNet
    elif model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
    else:
        print('** Please rechoose a model **\n-> Supported models are as follows:')
        print('   (1) AdaBoostRegressor / AdaBoostClassifier\n   (2) XGBRegressor / XGBClassifier')
        print('   (3) GradientBoostingRegressor / GradientBoostingClassifier\n   (4) HistGradientBoostingRegressor / HistGradientBoostingClassifier')
        print('   (5) RandomForestRegressor / RandomForestClassifier\n   (6) SVR / SVC')
        print('   (7) MLPRegressor / MLPClassifier\n   (8) ElasticNet / LogisticRegression')
        exit(1)

    model = joblib.load(filename_pkl + ".pkl")
    print('---------- Results based on the current loaded model ----------')
    print('> Current parameters:\n {}\n'.format(model.get_params()))

    return model, ML_type

def show_metrics(model, ML_type, y_test_pred, y_test_pred_proba, y_test, X_test):
    print("            >>>>  Metrics based on the best model  <<<<\n")
    if ML_type == "Classification":
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
        accuracy_test = accuracy_score(y_test, y_test_pred)
        print('> Accuracy on the test set:  {:.2%}'.format(accuracy_test))
        print('> Score on the test set:  {:.2%}'.format(model.score(X_test, y_test)))
        print('> Classification report on the test set:')
        print(classification_report(y_test, y_test_pred))

        roc_auc_test, average_precision_test = [], []
        for i in range(len(set(y_test))):
            roc_auc_test.append(roc_auc_score(y_test, y_test_pred_proba[:,i], multi_class='ovr'))
            average_precision_test.append(average_precision_score(y_test, y_test_pred_proba[:,i]))
        pd.set_option('display.float_format','{:12.6f}'.format)
        pd.set_option('display.colheader_justify', 'center')
        test_reports = pd.DataFrame(np.vstack((roc_auc_test, average_precision_test)).T, columns=['ROC-AUC','AP(PR-AUC)'])
        print('> Area under the receiver operating characteristic curve (ROC-AUC) and\n  average precision (AP) which summarizes a precision-recall curve as the weighted mean\n  of precisions achieved at each threshold on the test set:\n  {}\n'.format(test_reports))

    elif ML_type == "Regression":
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        print('> Mean squared error (MSE) on the test set:  {:.6f}'.format(mse_test))
        print('> Mean absolute error (MAE) on the test set:  {:.6f}'.format(mae_test))
        print('> R-squared (R^2) value on the test set:  {:.6f}\n'.format(model.score(X_test, y_test)))


###############  The ML training script starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
print('***  Scikit-learn evaluation ({0}) started at {1}  ***\n'.format(model_name, start_date.strftime("%Y-%m-%d %H:%M:%S")))

'''split training/test sets'''
if split_dataset:
    print('The dataset is splited into training and test sets, and therefore the target model will be evaluated on the test set...\n')
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)
else:
    print('The whole dataset will be used to evaluate the target model...\n')
    X_test, y_test = data_X, data_y

target_model, ML_type = load_model(model_name, filename_pkl)
y_test_pred = target_model.predict(X_test)
y_test_pred_proba = target_model.predict_proba(X_test) if ML_type == "Classification" else None
show_metrics(target_model, ML_type, y_test_pred, y_test_pred_proba, y_test, X_test)


end_time = time.time()
end_date = datetime.datetime.now()
print('***  Scikit-learn evaluation ({0}) terminated at {1}  ***\n'.format(model_name, end_date.strftime("%Y-%m-%d %H:%M:%S")))
total_running_time(end_time, start_time)


