import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from ml_metrics import rmse
# loading the data
Computerdata =pd.read_csv("D:\\ExcelR Data\\Assignments\\Multi linear Regration\\Computer_Data.csv")
Le = preprocessing.LabelEncoder() ##Label encoder() using for levels of categorical features into numerical values
Computerdata['Cd'] = Le.fit_transform(Computerdata['cd'])
Computerdata = Computerdata.drop('cd',axis = 1)
Computerdata['Multi'] = Le.fit_transform(Computerdata['multi'])
Computerdata = Computerdata.drop('multi',axis = 1)
Computerdata['Premium'] = Le.fit_transform(Computerdata['premium'])
Computerdata = Computerdata.drop('premium',axis = 1)
Computerdata.describe()
sns.pairplot(Computerdata)
Computerdata.columns
Computerdata.corr()#Correlation of coeficent
import statsmodels.formula.api as smf
#Building a model
#To predict the price of computers,here I'm  adding speed+hd+ram+screen+ads+trend+Cd+Multi+Premium against the Price
Model=smf.ols("price~speed+hd+ram+screen+ads+trend+Cd+Multi+Premium",data=Computerdata).fit()
Model.params
Model.summary()
#From my first model I got each and every variables as significant which means P-value less than 0.05 
#so here i'm predicting the price of computers from my model
Pred=Model.predict(Computerdata)
Pred
rootmse = rmse(Pred,Computerdata.Pr)
rootmse
