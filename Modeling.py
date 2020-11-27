import pandas as pd
import operator
from functools import reduce
from sklearn.utils import resample


print("----------Reading Data file and appending header-------------")
datasetheader=pd.read_csv('D:\Python\Mahesh\\field_names.txt',header=None)
headertranspose = datasetheader.transpose().values.tolist()
headertransposelist=reduce(operator.concat, headertranspose)
dataset=pd.read_csv('D:\Python\Mahesh\\breast-cancer.csv',names=headertransposelist) #combining data and header
print("----------Finding Mean and Median-------------")
MalignantDF =dataset.loc[dataset['diagnosis'] == 'M']
BenignDF =dataset.loc[dataset['diagnosis'] == 'B']
print('----------------------------------smoothness---------------------------------------')
print('Malignant smoothness mean:'+str(MalignantDF['smoothness_mean'].mean())+'\n')
print('Malignant smoothness median:'+str(MalignantDF['smoothness_mean'].median())+'\n')
print('Benign smoothness mean:'+str(MalignantDF['smoothness_mean'].mean())+'\n')
print('Benign smoothness median:'+str(MalignantDF['smoothness_mean'].median())+'\n')
print('-----------------------------------compactness----------------------------------')
print('Malignant compactness mean:'+str(MalignantDF['compactness_mean'].mean())+'\n')
print('Malignant compactness median:'+str(MalignantDF['compactness_mean'].median())+'\n')
print('Benign compactness mean:'+str(MalignantDF['compactness_mean'].mean())+'\n')
print('Benign compactness median:'+str(MalignantDF['compactness_mean'].median())+'\n')
print("----------Bootstarp Function-------------")
def bootstarpsample(df, sample_size):
    bootstrap_sample = pd.resample(df, replace=True, n_samples=sample_size, random_state=1)
    return (bootstrap_sample)
sampledf=bootstarpsample(dataset,1000)
print(sampledf)

print("---------------------Finding Feature importance with classification Decesion tree---------------------------")
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
datasettmp=dataset.copy()
y=datasettmp['diagnosis']
datasettmp.drop('ID', axis=1, inplace=True)
datasettmp.drop('diagnosis', axis=1, inplace=True)
X=datasettmp
model = DecisionTreeClassifier()
model.fit(X, y)
importance = model.feature_importances_
#feature vectore concavity_worst,fractal_dimension_mean,concave_points_mean
print( datasettmp.columns[20])
print( datasettmp.columns[27])
print( datasettmp.columns[21])
""""
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
"""
#plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
print('------------Model Building-----------------')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
def exercise_decesiontree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred_decesiontree = model.predict(X_test)
    acc_tree= accuracy_score(y_test, y_pred_decesiontree)
    return (acc_tree, model)
def exercise_logisticregression(X_train, X_test, y_train, y_test):
    clf_lr = LogisticRegression(solver = 'lbfgs')
    model = clf_lr.fit(X_train, y_train)
    importance = model.coef_[0]
    # summarize feature importance
    #symmetry_sd_error,symmetry_worst,area_worst
    print(X_train.columns[25])
    print(X_train.columns[26])
    print(X_train.columns[11])
    """
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    """
    y_pred_lr = clf_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    return(acc_lr,model)
def get_test_train(input_features,target,test_proportion):
    '''given set of input features, target, and proportion of holdback, return training and test sets'''
    X_train, X_test, y_train, y_test = train_test_split(input_features, target,\
                                                    test_size=test_proportion, random_state=42)
    return( X_train, X_test, y_train, y_test)
test_proportion = 0.2 # proportion of dataset to reserve for test
input_features=dataset.copy()
input_features.drop('ID', axis=1, inplace=True)
input_features.drop('diagnosis', axis=1, inplace=True)
target=dataset['diagnosis']
X_train, X_test, y_train, y_test = get_test_train(input_features,target,test_proportion)
print(exercise_decesiontree(X_train,X_test,y_train,y_test))
print(exercise_logisticregression(X_train,X_test,y_train,y_test))
print("end")