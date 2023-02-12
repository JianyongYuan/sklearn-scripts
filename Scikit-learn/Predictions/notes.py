#!/usr/bin/env python3
import pandas as pd
import numpy as np

df3 = pd.read_csv('prediction_results_3.csv').dropna(inplace=False)
df4 = pd.read_csv('prediction_results_4.csv').dropna(inplace=False)
df5 = pd.read_csv('prediction_results_5.csv').dropna(inplace=False)

sel_col1 = ['Smiles','IC50_GradientBoosting_Optuna_best','r_i_docking_score','r_glide_XP_GScore','r_i_glide_gscore']
sel_col2 = ['Smiles','IC50_GradientBoosting_Optuna_best','MOE_S']

df3 = df3[sel_col1]
df4 = df4[sel_col2]
df5 = df5[sel_col2]

print(df3.shape)
print(df4.shape)
print(df5.shape)


# cols1 = df1.columns
# cols2 = df2.columns
# print(cols1)
# print(cols2)
# print(cols1 == cols2)

# print(pd.merge(df1, df2, how='inner')) #查看两个df的交集

df345 = pd.concat([df3,df4,df5], axis=0).sort_values('IC50_GradientBoosting_Optuna_best',ascending=True,ignore_index=True)
df345_final = df345.loc[:, ~df345.columns.str.contains("^Unnamed")]
print(df345_final.shape)
print(df345_final)
df345_final.to_csv('regression_results_345.csv', encoding ='utf_8')

#exit()

df_reg = pd.read_csv('./IC50_RFE_RF20_regression/regression_results_345.csv')
df_cla = pd.read_csv('./IC50_RFE_RF20_classification/classification_results_345.csv')

sel_col_reg = ['Smiles','IC50_GradientBoosting_Optuna_best','r_i_docking_score','r_glide_XP_GScore','r_i_glide_gscore','MOE_S']
sel_col_cla = ['Smiles','IC50_RandomForest_Optuna_best','r_i_docking_score','r_glide_XP_GScore','r_i_glide_gscore','MOE_S']

df_reg = df_reg[sel_col_reg].drop_duplicates(keep='first',ignore_index=True)
df_cla = df_cla[sel_col_cla].drop_duplicates(keep='first',ignore_index=True)

# print(df_reg.shape)
# print(df_cla.shape)
print(df_reg)
print(df_cla)
all = pd.merge(df_reg, df_cla, how='inner', on=['Smiles','r_i_docking_score','r_glide_XP_GScore','r_i_glide_gscore','MOE_S'])
all.rename(columns={"IC50_GradientBoosting_Optuna_best":"IC50_regression","IC50_RandomForest_Optuna_best":"IC50<10_classification"},inplace=True)
all = all[['Smiles','r_i_docking_score','r_glide_XP_GScore','r_i_glide_gscore','MOE_S','IC50_regression','IC50<10_classification']]
print(all) #查看两个df的交集

#exit()
all.to_csv('summary_results_345.csv', encoding ='utf_8')


