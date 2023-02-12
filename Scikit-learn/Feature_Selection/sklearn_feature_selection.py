#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import datetime,time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn_pandas import DataFrameMapper

###############  Set required parameters here  ###############
task_type = 'Regression' # "Regression" or "Classification"
feature_selection_method = 'SelectFromModel_RF'  # "SelectKBest", "SelectPercentile", "RFE_RF", "RFECV_RF", "SelectFromModel_Tree", "SelectFromModel_RF"
num_feature_selected = 20
num_feature_selected_percentile = 50  # for "SelectPercentile" method
cv_fold = 3  # for "RFECV_RF" method
n_jobs =  6  # for "RFECV_RF" method
n_estimators = 150  # for "RFE_RF", "RFECV_RF" and "SelectFromModel_RF" method with RandomForest estimator

'''load the dataset'''
df = pd.read_csv('12.csv')
data_X = df.iloc[:,3:55]
data_y = df['IL1b_IC50']

# print(data_X[['dipole', 'FOSA', 'PISA', 'IP_HOMO', 'EA_LUMO', 'dip', 'ACxDN', 'glob', 'PSA', 'h_pKa', 'h_pKb', 'QPpolrz', 'QPlogPC16', 'QPlogPoct', 'QPlogPw', 'QPlogHERG', 'QPPCaco', 'QPPMDCK', 'QPlogKp', 'QPlogKhsa']])
# exit()


# print(data_y)
# exit()

# if task_type == 'Regression':
#     df = pd.read_csv('12.csv')
#     data_X = df.iloc[:,3:55]
#     data_y = df['IL1b_IC50']
# elif task_type == 'Classification':
#     from sklearn.datasets import load_iris
#     iris = load_iris()
#     data_X, data_y = pd.DataFrame(iris.data,columns=list(iris['feature_names'])), pd.Series(iris.target, name='flower')

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


###############  Some user-defined functions  ###############
def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time,2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))

def dataset_split(data_X, data_y, test_size, random_state):
    feature_names = data_X.columns
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=test_size, random_state=random_state)
    return X_train, y_train, feature_names

def result(select, feature_names, X_train):
    X_train_selected = select.transform(X_train)
    select_params = select.get_params()
    mask = select.get_support()
    selected_feature_name = feature_names[mask]

    print("\n>> Feature selection method: {}".format(feature_selection_method))
    print(">> X_train.shape: {}".format(X_train.shape))
    print(">> X_train_selected.shape: {}\n".format(X_train_selected.shape))
    print(">> Parameters for this estimator:\n {}\n".format(select_params))
    print(">> Total {0} feature names:\n {1}\n".format(len(feature_names), list(feature_names)))
    print(">> Selected {0} feature names:\n {1}\n".format(len(selected_feature_name), list(selected_feature_name)))

def univariate_selection(task_type, feature_selection_method):
    '''
    单变量特征选择

    单变量的特征选择是通过基于单变量的统计测试来选择最好的特征。
    它可以当做是评估器的预处理步骤。Scikit-learn
    将特征选择的内容作为实现了 transform 方法的对象：
    SelectKBest 移除那些除了评分最高的 K 个特征之外的所有特征
    SelectPercentile 移除除了用户指定的最高得分百分比之外的所有特征
    对每个特征应用常见的单变量统计测试: 假阳性率（false positive rate） 
    SelectFpr, 伪发现率（false discovery rate） SelectFdr ,
    或者族系误差（family wise error） SelectFwe 。
    GenericUnivariateSelect 允许使用可配置方法来进行单变量特征选择。
    它允许超参数搜索评估器来选择最好的单变量特征。

    更多信息参见：
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest

        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html?highlight=selectpercentile#sklearn.feature_selection.SelectPercentile
    '''

    if task_type == "Regression":
        from sklearn.feature_selection import mutual_info_regression
        score_func = mutual_info_regression
    elif task_type == "Classification":
        from sklearn.feature_selection import mutual_info_classif
        score_func = mutual_info_classif

    if feature_selection_method == "SelectKBest":
        from sklearn.feature_selection import SelectKBest
        select = SelectKBest(score_func, k=num_feature_selected)
    elif feature_selection_method == "SelectPercentile":
        from sklearn.feature_selection import SelectPercentile
        select = SelectPercentile(score_func, percentile=num_feature_selected_percentile)

    X_train, y_train, feature_names = dataset_split(data_X, data_y, test_size=0.2, random_state=0)
    select.fit(X_train,y_train)

    return select, feature_names, X_train

def RFE_selection(task_type, feature_selection_method):
    '''
    递归式特征消除(RFE)

    给定一个外部的估计器，可以对特征赋予一定的权重（比如，线性模型的相关系数），
    recursive feature elimination (RFE) 通过考虑越来越小的特征集合来递归的选择特征。
    首先，评估器在初始的特征集合上面训练并且每一个特征的重要程度是通过一个 coef_ 属性 或者 feature_importances_ 属性来获得。
    然后，从当前的特征集合中移除最不重要的特征。
    在特征集合上不断的重复递归这个步骤，直到最终达到所需要的特征数量为止。
    RFECV 在一个交叉验证的循环中执行 RFE 来找到最优的特征数量。
    这里递归的移除最不重要的像素点来对每个像素点（特征）进行排序

    更多信息参见：
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html?highlight=rfe#sklearn.feature_selection.RFE
    
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforestregressor#sklearn.ensemble.RandomForestRegressor

        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    '''

    if task_type == "Regression":
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=n_jobs)
    elif task_type == "Classification":  
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=0, n_jobs=n_jobs)
        
    if feature_selection_method == "RFE_RF":
        from sklearn.feature_selection import RFE
        select = RFE(estimator, n_features_to_select=num_feature_selected, verbose=1)
    elif feature_selection_method == "RFECV_RF":
        from sklearn.feature_selection import RFECV
        select = RFECV(estimator, min_features_to_select=num_feature_selected, verbose=1, cv=cv_fold, n_jobs=n_jobs)

    X_train, y_train, feature_names = dataset_split(data_X, data_y, test_size=0.2, random_state=0)
    select.fit(X_train, y_train)

    return select, feature_names, X_train

def SFM_selection(task_type, feature_selection_method):
    '''
    基于Tree树的特征选取
 
    基于树的 estimators （查阅 sklearn.tree 模块和树的森林 在 sklearn.ensemble 模块）
    可以用来计算特征的重要性，然后可以消除不相关的特征
    （当与 sklearn.feature_selection.SelectFromModel 等元转换器一同使用时）

    更多信息参见：
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html?highlight=selectfrommodel#sklearn.feature_selection.SelectFromModel

        https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier

        https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html#sklearn.tree.ExtraTreeRegressor

        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforestregressor#sklearn.ensemble.RandomForestRegressor

        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

    '''

    from sklearn.feature_selection import SelectFromModel

    if task_type == "Regression":
        if feature_selection_method == "SelectFromModel_Tree":
            from sklearn.ensemble import ExtraTreesRegressor
            estimator = ExtraTreesRegressor(n_jobs=n_jobs)
        elif feature_selection_method == "SelectFromModel_RF":
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=n_jobs)
    elif task_type == "Classification":  
        if feature_selection_method == "SelectFromModel_Tree":
            from sklearn.ensemble import ExtraTreesClassifier
            estimator = ExtraTreesClassifier(class_weight='balanced', n_jobs=n_jobs)
        elif feature_selection_method == "SelectFromModel_RF":
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=0, n_jobs=n_jobs)

    select = SelectFromModel(estimator, max_features=num_feature_selected)
    
    X_train, y_train, feature_names = dataset_split(data_X, data_y, test_size=0.2, random_state=0)
    select.fit(X_train, y_train)

    return select, feature_names, X_train


###############  The feature selection for regression task starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
print('***  Scikit-learn feature selection for {0} started at {1}  ***\n'.format(task_type.lower(), start_date.strftime("%Y-%m-%d %H:%M:%S")))

if feature_selection_method in ["SelectKBest", "SelectPercentile"]:
    select, feature_names, X_train = univariate_selection(task_type, feature_selection_method)
elif feature_selection_method in ["RFE_RF", "RFECV_RF"]:
    select, feature_names, X_train = RFE_selection(task_type, feature_selection_method)
elif feature_selection_method in ["SelectFromModel_Tree", "SelectFromModel_RF"]:
    select, feature_names, X_train = SFM_selection(task_type, feature_selection_method)

result(select, feature_names, X_train)

end_time = time.time()
end_date = datetime.datetime.now()
print('***  Scikit-learn feature selection for {0} terminated at {1}  ***\n'.format(task_type.lower(), end_date.strftime("%Y-%m-%d %H:%M:%S")))
total_running_time(end_time, start_time)





