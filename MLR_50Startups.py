# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:54:29 2020

@author: LENOVO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from ml_metrics import rmse
#Loading the data
Startups=pd.read_csv("D:\\ExcelR Data\\Assignments\\Multi linear Regration\\50_Startups.csv")
Le = preprocessing.LabelEncoder() ##Label encoder() using for levels of categorical features into numerical values
Startups['st'] = Le.fit_transform(Startups['St'])
Startups = Startups.drop('St',axis = 1)
Startups.columns
Startups.describe() #To find count,mean,std,min etc

Startups.corr() #correlation of coeficent [Profit-R&D have 0.97 which is >0.85]
# Normalize the Data
Startups_normal=preprocessing.normalize(Startups)
# visualization
plt.hist(Startups.Rd)
plt.boxplot(Startups.Ad)
plt.hist(Startups.Ms)
plt.boxplot(Startups.st)
sns.pairplot(Startups) #pair plot
import statsmodels.formula.api as smf
#Building a Model
#To Pridict THe Profit of 50-Startups im Adding (RD_Spend+Administration+Marketing_Spend+state)
Model1=smf.ols("Pr~Rd+Ad+Ms+st",data=Startups).fit()
Model1.params
Model1.summary()
pred1=Model1.predict(Startups)
pred1
#Now by Summary we can see that Administration,Marketing_Spend,state all Veriables have P-v >0.05 so compare with o/p indiviguly
#only with Administration
Model_a=smf.ols("Pr~Ad",data=Startups).fit()
Model_a.summary()#0.162
# p-value <0.05 .. It is significant 

#only with Marketing_Spend
Model_m=smf.ols("Pr~Ms",data=Startups).fit()
Model_m.summary()#0.000

#only with state
Model_s=smf.ols("Pr~st",data=Startups).fit()
Model_s.summary()#0.482


#Marketing_Spend & state-insignificant P-value,Administration & state both have insignificant P-value 
# So there may be a chance of considering only Administration & Marketing_Spend by droping state

# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(Model)

Startups_new = Startups.drop(Startups.index[[45,46,49,48,6,19]],axis=0) 

# Preparing model                  
Model2 =smf.ols("Pr~Rd+Ad+Ms",data=Startups).fit()
Model2.params
Model2.summary()

pred2=Model.predict(Startups_new)
pred2

# calculating VIF's values of independent variables
rsq_Adm = smf.ols('Ad~Ms',data=Startups_new).fit().rsquared  
vif_Adm = 1/(1-rsq_Adm) #1.007217

rsq_st = smf.ols('Ms~Ad',data=Startups_new).fit().rsquared  
vif_st = 1/(1-rsq_st)#1.007217

# Storing vif values in a data frame
d1 = {'Variables':['Ad','Ms'],'VIF':[vif_Adm,vif_st]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

Model_a =smf.ols("Pr~Ad",data=Startups).fit()
Model_a.summary()

Model_m=smf.ols("Pr~Ms",data=Startups).fit()
Model_m.summary()

model_am=smf.ols("Pr~Ad+Ms",data=Startups).fit()
model_am.summary()

# Added varible plot 
sm.graphics.plot_partregress_grid(Model2)

#Final Model
Model3=smf.ols("Pr~Rd+Ms",data=Startups).fit()
Model3.params
Model3.summary()

pred3=Model.predict(Startups_new)
pred3

#Finally i'm going for Root mean square error(RMSE) to check the average error in my data set
rootmse = rmse(pred3,Startups_new.Pr)
rootmse
Actual=Startups_new.Pr
#Creating a dataframe set for actual and predicted price
df = pd.DataFrame(list(zip(pred3, Actual)),columns =['Predicted Prices', 'Actual Prices'])
#Next i'm going for to create a r^2 value table for my three models
values = list([Model1.rsquared,Model2.rsquared,Model3.rsquared])#R^2 values
coded_variables = list(['Model1.rsquared','Model2.rsquared','Model3.rsquared'])#
variables = list(['Model 1','Model 2','Model 3'])
Rsquared_model = pd.DataFrame(list(zip(variables,coded_variables,values)),columns = ['Models','Variabels Named in the code','R^Squared Values'])
Rsquared_model

   Models       Variabels Named in the code   R^Squared Values
0  Model 1             Model1.rsquared          0.950746
1  Model 2             Model2.rsquared          0.950746
2  Model 3             Model3.rsquared          0.950450