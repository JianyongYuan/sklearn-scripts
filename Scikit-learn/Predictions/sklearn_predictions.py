#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import pandas as pd
import numpy as np
import itertools as it
import os,datetime,time,joblib

###############  Set required parameters here  ###############
input_filename = [  'HistGradientBoosting_Optuna_best'
                ]  # the filename.pkl files in the current working directory to be load as target models
output_filename = 'prediction_IC50_regression' # the results are saved to the *.csv file
sorting_choice = 'ascending' # "ascending" or "descending"

'''load datasets'''
selected_features = ['MolWt','NumRotatableBonds','AromaticProportion']
df = pd.read_csv('../../MolLogP_dataset.csv')
data_X = df[selected_features]
data_X = df[selected_features].dropna(inplace=False)
# print(data_X)
# exit()

'''set target Y ranges; 'all' for all ranges'''
# y_ranges = "(data_X_y['Hardness_EN_best'] > 60.5)&(data_X_y['Hardness_EN_best'] < 61)&(data_X_y['MA300_EN_best'] > 1)"
y_ranges = "all"

###############  Some user-defined functions and variables  ###############
sorting = True if sorting_choice == 'ascending' else False

def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time,2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))

def model_predictions(input_filename):
    filename = input_filename + ".pkl"
    model = joblib.load(filename) 
    y_pred = model.predict(data_X)
    data_y = pd.DataFrame(y_pred, columns=[input_filename])
    data_X_y = pd.concat([df['Smiles'], data_X, data_y], axis=1)
    data_X_y_sorted = data_X_y.sort_values(input_filename, ascending=sorting, ignore_index=True)

    print('---------- Information of the {0} model ----------'.format(filename))
    print('> Parameters of the {0} model:\n {1}\n'.format(filename, model.get_params()))
    print('> Results based on the {0} model:\n {1}\n'.format(filename, data_X_y_sorted))
    return y_pred

def get_results(y_preds, data_X, input_filename, y_ranges='all'):
    print('\n-------->> FINAL RESULTS BASED ON THE ABOVE {0} MODEL(S) <<--------'.format(len(input_filename)))
    print('> Target Y ranges:\n  {0}\n'.format(y_ranges))
    # pd.set_option('max_colwidth', 50)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.width', 1000)
    data_X_y = pd.concat([data_X, y_preds], axis=1)
    data_X_y = pd.concat([df['Smiles'], data_X_y], axis=1)  #增加smiles与docking score
    # data_X_y = pd.concat([df['Smiles'], data_X_y, df[['r_i_docking_score','r_glide_XP_GScore','r_i_glide_gscore','MOE_S']]], axis=1)  #增加smiles与docking score

    y_ranges = eval(y_ranges)
    final_results = data_X_y[y_ranges] if type(y_ranges).__name__ == "Series" else data_X_y

    final_results = final_results.sort_values(input_filename, ascending=sorting, ignore_index=True)
    final_results = final_results if len(final_results.index) != 0 else ' @@ No record falls in the target Y ranges! @@'
    print('> Final results shown in {0} orders of target Y columns:\n {1}\n'.format(sorting_choice, final_results))

    
    if type(final_results).__name__ == "DataFrame":
        filename = output_filename + ".csv"
        final_results.to_csv(filename, encoding ='utf_8')
        print('~~ The results are saved as the \'{0}\' file in the current working directory ~~'.format(filename))
        print('   CWD: {0}\n'.format(os.getcwd()))


###############  The ML prediction script starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
print('***  Scikit-learn predictions (in target Y ranges) started at {0}  ***\n'.format(start_date.strftime("%Y-%m-%d %H:%M:%S")))


y_preds = []
for i in input_filename:
    y_preds.append(model_predictions(i)[:,np.newaxis])
y_preds = pd.DataFrame(np.hstack(y_preds), columns=input_filename)
get_results(y_preds,data_X,input_filename,y_ranges)


end_time = time.time()
end_date = datetime.datetime.now()
print('***  Scikit-learn predictions (in target Y ranges) terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
total_running_time(end_time,start_time)





