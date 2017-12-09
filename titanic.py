import pandas as pd
import string
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from pandas.plotting import scatter_matrix
%matplotlib inline
import numpy as np
from sklearn.model_selection import GridSearchCV

def process(df):
    char1 = ', '
    char2 = '.'
    titles = []
    title = []
    mylist = list(df['Name'])
    for name in mylist:
        title.append(name[name.find(char1)+1 : name.find(char2)])
        substr = name[name.find(char1)+1 : name.find(char2)]
        if substr not in titles:
            titles.append(substr)
    titles = [x.strip(' ') for x in titles]
    title = [x.strip(' ') for x in title]
    new_titles = []
    
    for name in title:
        if name in ['Dr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Mr']:
            new_titles.append('Mr')
        elif name in ['Countess', 'Mme','Mrs']:
            new_titles.append('Mrs')
        elif name in ['Mlle', 'Ms','Miss']:
            new_titles.append('Miss')
        else:
            new_titles.append('Master')
    
    df['title'] = pd.DataFrame(new_titles)
    df = df.join(pd.get_dummies(df['title']))
    
    df.drop(['Name','PassengerId','Ticket','title'],inplace=True,axis=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df = df.join(pd.get_dummies(df['Embarked']))
    df = df.join(pd.get_dummies(df['Sex']))
    df['is_cabin'] = 0
    df.loc[pd.isnull(df['Cabin']) == True,'is_cabin'] = 0
    df.loc[pd.isnull(df['Cabin']) == False,'is_cabin'] = 1
    df = df.drop(['Embarked','Sex','Cabin'],axis=1)
    df['family_size'] = df['SibSp'] + df['Parch']
    df.drop(['SibSp','Parch'],inplace=True,axis=1)
    df['is_rich'] = 1
    df.loc[df['Fare'] >= df['Fare'].median(), 'is_rich'] = 1
    df.loc[df['Fare'] < df['Fare'].median(), 'is_rich'] = 0
    df.loc[df['Age'] >= 18,'is_child'] = 1
    df.loc[df['Age'] < 18,'is_child'] = 0
    df.drop(['Age','Fare'],inplace=True,axis=1)
    return df

df = pd.read_csv("train.csv")
ids = df['PassengerId']
df_test = pd.read_csv('test.csv')

df = process(df)
pred = df['Survived']
df.drop(['Survived'],inplace=True,axis=1)
df_test = process(df_test)

svc = svm.SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 100],'tol':[1e-7,1e-1],'gamma':[1e-5,10000]}
clf1 = GridSearchCV(svc, parameters)
clf1.fit(df,pred)
df_result = pd.DataFrame(clf1.predict(df_test))
df_result.to_csv('result.csv',index=True)
