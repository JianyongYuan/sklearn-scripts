#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import os,time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

'''input basic settings'''
classification_label = None # None for regression tasks; 0, 1, 2, etc for classification tasks
plot_type = "summary" # "importance", "summary", "force", "waterfall", "dependence", and "interaction"
show_fig = True  # whether to show figures after plotting results
sample_index = 1  # only for the force and waterfall plots with the specific sample, "0", "1", "2", etc.
feature_dependence = ['NumRotatableBonds','AromaticProportion']  # only for the dependence plot with specific two features, e.g.:['DS','R1502']
feature_interaction = [] # only for the interaction plot with the specific two features, e.g.:['DC01T','DC01T']; empty list '[]' for all features
max_display = 10  # set the number of features to be displayed
model_type = "regression" if classification_label == None else "classification"

'''load pipline'''
filename = "reg0"  # input the name of the *.pkl file
pipeline = joblib.load(filename + ".pkl") 
model = pipeline._final_estimator

'''load dataset'''
selected_features = ['MolWt','NumRotatableBonds','AromaticProportion']
df = pd.read_csv('MolLogP_dataset.csv')
data_X = df[selected_features]
data_y = df['MolLogP']


'''select data'''
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

# X_background = data_X    # background data samples (e.g. training set)
# X_selected = data_X      # selected data samples to be plotted (e.g. test set)

X_background = X_train.sample(n=200,random_state=0)     # background data samples (e.g. training set)
X_selected = X_test.sample(n=50,random_state=0)        # selected data samples to be plotted (e.g. test set)

# print(X_selected)
# exit()


'''define useful functions'''
def pipeline_shap(pipeline, X_background, X_selected, KernelExplainer=False, interaction=False, model_type="regression", label=None):
    model = pipeline._final_estimator
    feature_values = X_selected
    sort_cols, onehot_cols = [], []
    OneHot_flag = False
    Select_flag = False

    if "ct" in pipeline.named_steps:
        col_trans_flag = True
        mapper = pipeline.named_steps["ct"]
    else:
        col_trans_flag = False

    # sort columns according to the DataFrameMapper or ColumnTransformer
    if col_trans_flag:
        if "DataFrameMapper" in str(mapper):
            ct_name = "DataFrameMapper"
            colname_iloc = 0
            all_transformers = mapper.features
        elif "ColumnTransformer" in str(mapper):
            ct_name = "ColumnTransformer"
            colname_iloc = 2
            all_transformers = mapper.transformers_
        print("Info: The {0} module is employed to preprocess data...\n".format(ct_name))

        for i in all_transformers:
            if 'Select' in str(i[1]):
                Select_flag = True
                inp_col = i[colname_iloc]
                for j in i[1]:
                    if 'Select' in str(j):
                        sel_col_index = j.get_support(indices=True)
                        break
                sel_cols = [inp_col[index] for index in sel_col_index]
                sort_cols += sel_cols
            else:
                sort_cols += i[colname_iloc]
            if 'OneHot' in str(i[1]):
                OneHot_flag = True
                if 'Select' in str(i[1]):
                    onehot_cols += sel_cols
                else:
                    onehot_cols += i[colname_iloc]
   
        feature_values = feature_values[sort_cols]
        print("Sorted feature values of the first five samples:\n{0}\n".format(feature_values.head()))

        if Select_flag:
            feature_values_mapper = mapper.transform(X_selected)
            print("Info: The Select* transformer is detected! The number of sorted features may be reduced...\n")
        else:
            feature_values_mapper = mapper.transform(feature_values)
        print("Mapped feature values of the first five samples:\n{0}\n".format(feature_values_mapper[:5,:]))

        if OneHot_flag:
            print("Info: The OneHotEncoder is detected! Therefore, merge the mapped shap values...\n")

    # select the explainer module and obtain expected_values
    if KernelExplainer:
        if col_trans_flag:
            if model_type == "regression":
                explainer = shap.KernelExplainer(model.predict, mapper.transform(X_background))
            elif model_type == "classification":
                f = lambda x: model.predict_proba(x)[:,label]
                explainer = shap.KernelExplainer(f, mapper.transform(X_background))
        else:
            if model_type == "regression":
                explainer = shap.KernelExplainer(model.predict, X_background)
            elif model_type == "classification":
                f = lambda x: model.predict_proba(x)[:,label]
                explainer = shap.KernelExplainer(f, X_background)
        expected_values = explainer.expected_value
    else:
        if model_type == "regression":
            explainer = shap.Explainer(model)
            expected_values = explainer.expected_value[0]
        elif model_type == "classification":
            f = lambda x: model.predict_proba(x)[:,label]
            if col_trans_flag:
                X_background_mapper = mapper.transform(X_background)
                med = X_background_mapper.median().values.reshape((1, X_background_mapper.shape[1]))
            else:
                med = X_background.median().values.reshape((1, X_background.shape[1]))
            explainer = shap.Explainer(f, med)


    # calculate shap_values or shap_interaction_values
    if interaction == True and col_trans_flag == True:
        mapper_shap_values = explainer.shap_interaction_values(feature_values_mapper)
        total_samples = len(mapper_shap_values)
        shap_values = []
        index_map = []

        index = 0
        for col in sort_cols:
            if col in onehot_cols:
                col_index_span = len(X_background[col].unique())
            else:
                col_index_span = 1
            index_map.append((index, col_index_span))
            index += col_index_span

        for s in range(total_samples):
            shap_values_one_sample = np.zeros((len(sort_cols),len(sort_cols)))
            for i in range(len(sort_cols)):
                for j in range(len(sort_cols)):
                    shap_values_one_sample[i,j] = mapper_shap_values[s][
                        index_map[i][0]:index_map[i][0]+index_map[i][1], index_map[j][0]:index_map[j][0]+index_map[j][1]
                         ].sum()
            shap_values.append(shap_values_one_sample)
        shap_values = np.array(shap_values)
         
    elif interaction == False and col_trans_flag == True:
        if model_type == "classification" and KernelExplainer == False:
            mapper_shap_values = explainer(feature_values_mapper).values
            expected_values = explainer(feature_values_mapper).base_values[0]
        else:
            mapper_shap_values = explainer.shap_values(feature_values_mapper)

        shap_values = pd.DataFrame(index=feature_values.index, columns=feature_values.columns)
        col_index = 0
        for col in sort_cols:
            if col in onehot_cols:
                col_index_span = len(X_background[col].unique())
                shap_values[col] = mapper_shap_values[:, col_index: col_index + col_index_span].sum(1)
                col_index += col_index_span
            else:
                shap_values[col] = mapper_shap_values[:, col_index]
                col_index += 1

    elif interaction == True and col_trans_flag == False:
        shap_values = explainer.shap_interaction_values(feature_values)

    elif interaction == False and col_trans_flag == False:
        if model_type == "classification" and KernelExplainer == False:
            shap_values = explainer(feature_values).values
            expected_values = explainer(feature_values).base_values[0]
        else:
            shap_values = explainer.shap_values(feature_values)
        shap_values = pd.DataFrame(shap_values, index=feature_values.index, columns=feature_values.columns)
        

    return feature_values, shap_values, expected_values

def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time,2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print("Info: Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s)".format(int(days),int(hours),int(minutes),seconds))
    
def input_check():
    if model_type == "classification":
        if not isinstance(classification_label, int):
            print("Error: Please input an integer in the 'classification_label' field!")
            exit()

    if plot_type not in ['importance', 'summary', 'force', 'waterfall', 'dependence', 'interaction']:
        print("Error: Please input one of the following plot types:")
        print("       'importance', 'summary', 'force', 'waterfall', 'dependence', or 'interaction'")
        exit()
    elif plot_type in ['force', 'waterfall']:
        if not isinstance(sample_index, int): 
            print("Error: Please input an integer in the 'sample_index' field!")
            exit()
    elif plot_type == "dependence":
        if not isinstance(feature_dependence, list) or len(feature_dependence) != 2: 
            print("Error: Please input a list with two elements in the 'feature_dependence' field!")
            exit()
    elif plot_type == "interaction":
        if not isinstance(feature_interaction, list) or len(feature_interaction) not in [0,2]: 
            print("Error: Please input an empty list or a list with two elements in the 'interaction' field!")
            exit()


'''creat explainer and get shap_values and expected_values'''
input_check()
start_time = time.time()

try :
    shap.Explainer(model)
    print("Use the smart Explainer to analyse the target model...\n")
    kernel_flag = False
    if plot_type == "interaction":
        if model_type == "classification":
            pass
        elif model_type == "regression":
            feature_values, shap_interaction_values, expected_values = pipeline_shap(pipeline, X_background, X_selected, KernelExplainer=False, interaction=True, model_type=model_type, label=classification_label)
    else:
        feature_values, shap_values, expected_values = pipeline_shap(pipeline, X_background, X_selected, KernelExplainer=False, interaction=False, model_type=model_type, label=classification_label)
except TypeError as e:
    print("Warning: {0}\nTherefore, use the KernelExplainer instead to analyse the target model...\n".format(e))
    kernel_flag = True
    if plot_type == "interaction":
        pass
    else:
        feature_values, shap_values, expected_values = pipeline_shap(pipeline, X_background, X_selected, KernelExplainer=True, interaction=False, model_type=model_type, label=classification_label)

'''obtain results and plots'''
save_plot_name = filename + "_" + plot_type + ".png"

if plot_type == "importance":
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = shap_values.columns
    feature_importance['importance'] = feature_importance['feature'].map(np.abs(shap_values).mean(0))
    feature_importance = feature_importance.sort_values('importance', ascending=False)[:max_display]
    print("Feature importance:\n{0}\n".format(feature_importance))
    shap.summary_plot(shap_values.values, feature_values, plot_type="bar", show=False, max_display=max_display)
    for x,y in enumerate(feature_importance.sort_values('importance', ascending=True).values[-max_display:]):
        plt.text(y[1], x-0.1, '%.4f' %round(y[1],4), ha='left')

elif plot_type == "summary":
    shap.summary_plot(shap_values.values, feature_values, show=False, max_display=max_display)

elif plot_type == "force" or plot_type == "waterfall":
    if model_type == "regression":
        y_pred = pipeline.predict(X_selected)
    elif model_type == "classification":
        y_pred = pipeline.predict_proba(X_selected)[:,classification_label]
    print("Target sample:\n{0}\n".format(feature_values.iloc[sample_index,:]))
    print("Expected [base] value:   {:.4f}".format(expected_values))
    print("Predicted [f(x)] value:  {:.4f}\n".format(y_pred[sample_index]))
    if plot_type == "force":
        shap.force_plot(expected_values, shap_values.values[sample_index,:], feature_values.iloc[sample_index,:], matplotlib=True, show=False)
    elif plot_type == "waterfall":
        shap.waterfall_plot(shap.Explanation(values=shap_values.values[sample_index,:], base_values=expected_values, data=feature_values.iloc[sample_index,:]), max_display=max_display, show=False)

elif plot_type == "interaction":
    if kernel_flag == False:
        if model_type == "classification":
            print("\nError: The smart Explainer (classification) cannot output the interaction plot (shap_interaction_values)!")
            show_fig = None
        elif model_type == "regression":
            if feature_interaction:
                shap.dependence_plot(tuple(feature_interaction), shap_interaction_values, feature_values, show=False)
                save_plot_name = filename + "_main_effect.png" if feature_interaction[0] == feature_interaction[1] else     filename + "_target_interactions.png"
            else:
                shap.summary_plot(shap_interaction_values, feature_values, plot_type="compact_dot", max_display=2*max_display,  show=False)
                save_plot_name = filename + "_top" + str(2*max_display) + "_interactions.png"
    else:
        print("\nError: The KernelExplainer cannot output the interaction plot (shap_interaction_values)!")
        show_fig = None

elif plot_type == "dependence":
    shap.dependence_plot(feature_dependence[0], shap_values.values, feature_values, interaction_index=feature_dependence[1], show=False)


'''output target figures'''
end_time = time.time()
if show_fig != None:
    plt.savefig(save_plot_name, bbox_inches='tight', dpi=300)
    print('Info: The target graph \'{}\' is saved in the following path:'.format(save_plot_name))
    print('      {0}\n'.format(os.getcwd() + os.sep + save_plot_name))
    total_running_time(end_time, start_time)
    
    if show_fig == True:
        plt.show()



