# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:57:01 2020

@author: LENOVO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from ml_metrics import rmse
#Loading the data
ToyotaCorl=pd.read_csv("D:\\ExcelR Data\\Assignments\\Multi linear Regration\\ToyotaCorolla.csv",encoding='unicode_escape') #unicode_escape is for producing an aski encoding 
ToyotaCorl.columns
ToyotaCorl = ToyotaCorl.drop(['Model','Mfg_Month','Mfg_Year','Fuel_Type','Met_Color','Color','Automatic','Cylinders','Mfr_Guarantee','BOVAG_Guarantee','Guarantee_Period','ABS','Airbag_1','Airbag_2','Airco','Automatic_airco','Boardcomputer','CD_Player','Central_Lock','Powered_Windows','Power_Steering','Radio','Mistlamps','Sport_Model','Backseat_Divider','Metallic_Rim','Radio_cassette','Tow_Bar'],axis = 1)
#Here im removing all unwanted variables from data frame
ToyotaCorl.columns
ToyotaCorl.describe() #To find count,mean,std,min etc

ToyotaCorl.corr() #correlation of coeficent

ToyotaCorl_normal=preprocessing.normalize(ToyotaCorl)
# visualization
plt.hist(ToyotaCorl.Pr)
plt.hist(ToyotaCorl.Age)
plt.hist(ToyotaCorl.KM)
plt.hist(ToyotaCorl.HP)
plt.hist(ToyotaCorl.cc)
plt.hist(ToyotaCorl.Drs)
plt.hist(ToyotaCorl.Grs)
plt.hist(ToyotaCorl.Qt)
plt.hist(ToyotaCorl.Wt)
sns.pairplot(ToyotaCorl) #pair plot

#Building a Model
#To Predicting the price
import statsmodels.formula.api as smf
Model1=smf.ols("Pr~Age+KM+HP+cc+Drs+Grs+Qt+Wt",data=ToyotaCorl).fit()
Model1.params
Model1.summary()
pred1=Model1.predict(ToyotaCorl)
pred1
#From first model only cc and Drs both have higher p-value greater then 0.05
#p-value of cc-0.179 and Drs-0.968 
#Next going for significance
Model_c=smf.ols("Pr~cc",data=ToyotaCorl).fit()
Model_c.summary()#0.000

Model_d=smf.ols("Pr~Drs",data=ToyotaCorl).fit()
Model_d.summary()#0.000

Model_d=smf.ols("Pr~cc+Drs",data=ToyotaCorl).fit()
Model_d.summary()
#seems like there is no issue  for pvalue of cc+Doors against Price


# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(Model1)

ToyotaCorl_new = ToyotaCorl.drop(ToyotaCorl.index[[80,960,221]],axis=0) 

# Preparing model                  
Model2 =smf.ols("Pr~Age+KM+HP+cc+Drs+Grs+Qt+Wt",data=ToyotaCorl_new).fit()
Model2.params
Model2.summary()

pred2=Model2.predict(ToyotaCorl_new)
pred2
#from this model2 i can say my gear variable is having p-value(0.010) which is greater than 0.05.There for i'm going for significance
Mode2_d=smf.ols("Pr~Grs",data=ToyotaCorl).fit()
Mode2_d.summary()

sm.graphics.influence_plot(Model2)

ToyotaCorl_new = ToyotaCorl_new.drop(ToyotaCorl_new.index[[109,111,110,956,991,654,601]],axis=0)

#Building 3rd model
Model3 =smf.ols("Pr~Age+KM+HP+cc+Drs+Grs+Qt+Wt",data=ToyotaCorl_new).fit()
Model3.params
Model3.summary()

pred3=Model3.predict(ToyotaCorl_new)
pred3
#Now from model3 we can see that all the variables have the significant p-values
#To improve my R^2 value im going for logerithemic Transformation
Model4 =smf.ols("np.log(Pr)~Age+KM+HP+cc+Drs+Grs+Qt+Wt",data=ToyotaCorl_new).fit()
Model4.summary()
#from model4 i can see the 'doors' having higher pvalue(0.125) which is greater than 0.05
#So remove that variable 'Drs'
pred4=Model4.predict(ToyotaCorl_new)
pred4

ToyotaCorl_new = ToyotaCorl_new.drop(["Drs"],axis=1)

Model5 =smf.ols("Pr~Age+KM+HP+cc+Grs+Qt+Wt",data=ToyotaCorl_new).fit()
Model5.summary()

pred5=Model5.predict(ToyotaCorl_new)
pred5
#from model5 i can see pvalues for every variables which is less than 0.05 and R^2 value i got is 0.853
#again going for influence plot to improve my R^2 value
sm.graphics.influence_plot(Model5)

ToyotaCorl_new = ToyotaCorl.drop(ToyotaCorl.index[[109,601,956,991]],axis=0)

Model6 =smf.ols("np.log(Pr)~Age+KM+HP+cc+Grs+Qt+Wt",data=ToyotaCorl_new).fit()
Model6.summary()

pred6=Model6.predict(ToyotaCorl_new)
pred6
# Added varible plot 
sm.graphics.plot_partregress_grid(Model6)

#Finally i'm going for Root mean square error(RMSE) to check the average error in my data set
rootmse = rmse(pred6,ToyotaCorl_new.Pr)
rootmse
Actual=ToyotaCorl_new.Pr

#Here i'm bulding a table to see which model having higher R^2 value
values = list([Model1.rsquared,Model2.rsquared,Model3.rsquared,Model4.rsquared,Model5.rsquared,Model6.rsquared])
coded_variables = list(['Model1.rsquared','Model2.rsquared','Model3.rsquared','Model4.rsquared','Model5.rsquared','Model6.rsquared'])
variables = list(['Model 1','Model 2','Model 3','Model 4','Model 5','Model 6'])
Rsquared_model = pd.DataFrame(list(zip(variables,coded_variables,values)),columns = ['Models','Variabels Named in the code','R^Squared Values'])
Rsquared_model

    Models      Variabels Named in the code  R^Squared Values
0  Model 1             Model1.rsquared          0.863763
1  Model 2             Model2.rsquared          0.885185
2  Model 3             Model3.rsquared          0.878933
3  Model 4             Model4.rsquared          0.853363
4  Model 5             Model5.rsquared          0.878149
5  Model 6             Model6.rsquared          0.852216