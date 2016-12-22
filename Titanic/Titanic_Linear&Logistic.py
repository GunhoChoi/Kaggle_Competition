import csv as csv
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


#read file and save as pandas dataframe
csv_file_object = csv.reader(open('train.csv', 'rb'))      
header = csv_file_object.next()                            
data=[]                            

for row in csv_file_object:                 
    data.append(row)                      

data=pd.DataFrame(data,columns=header)
#print(data.loc[data['Fare']>=0]["Survived"])
#print(header)

#preprocessing data
actual_result=data["Survived"]
input_data=data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
size=input_data.shape[0]

#sex
x=input_data.loc[input_data["Sex"]=='male'].index
for i in range(size):
	if i in x:
		input_data.set_value(i,"Sex",0)
	else:
		input_data.set_value(i,"Sex",1)

#Embark
s=input_data.loc[input_data["Embarked"]=='S'].index
c=input_data.loc[input_data["Embarked"]=='C'].index
for i in range(size):
	if i in s:
		input_data.set_value(i,"Embarked",0)
	elif i in c:
		input_data.set_value(i,"Embarked",1)
	else:
		input_data.set_value(i,"Embarked",2)

#str to numeric
input_data=input_data.convert_objects(convert_numeric=True)

#age
mean=input_data.mean()
zero_age=input_data.loc[input_data["Age"].isnull()].index
for i in zero_age:
	input_data.set_value(i,"Age",int(mean["Age"]))

input_data["Age"]-=mean["Age"]
input_data["Fare"]-=mean["Fare"]
input_data["Age"]/=(max(input_data["Age"])-min(input_data["Age"]))
input_data["Fare"]/=(max(input_data["Fare"])-min(input_data["Fare"]))

#linear regression
regr = linear_model.LinearRegression(normalize=True)
model1=regr.fit(input_data, actual_result)
linear_score=regr.score(input_data, actual_result)
print(linear_score)

#logistic regression
lg = LogisticRegression(penalty='l2')
model2=lg.fit(input_data, actual_result)
logistic_score=lg.score(input_data, actual_result)
print(logistic_score)

###################################################################################

#read file and save as pandas dataframe
csv_file_object = csv.reader(open('test.csv', 'rb'))      
header = csv_file_object.next()                            
data=[]                            

for row in csv_file_object:                 
    data.append(row)                      

data=pd.DataFrame(data,columns=header)
#print(data.loc[data['Fare']>=0]["Survived"])
#print(header)

#preprocessing data
#actual_result=data["Survived"]
input_data=data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
size=input_data.shape[0]

#sex
x=input_data.loc[input_data["Sex"]=='male'].index
for i in range(size):
	if i in x:
		input_data.set_value(i,"Sex",0)
	else:
		input_data.set_value(i,"Sex",1)

#Embark
s=input_data.loc[input_data["Embarked"]=='S'].index
c=input_data.loc[input_data["Embarked"]=='C'].index
for i in range(size):
	if i in s:
		input_data.set_value(i,"Embarked",0)
	elif i in c:
		input_data.set_value(i,"Embarked",1)
	else:
		input_data.set_value(i,"Embarked",2)

#str to numeric
input_data=input_data.convert_objects(convert_numeric=True)

#age
mean=input_data.mean()
zero_age=input_data.loc[input_data["Age"].isnull()].index
for i in zero_age:
	input_data.set_value(i,"Age",int(mean["Age"]))

#print mean["Fare"]

f=input_data.loc[input_data["Fare"].isnull()].index
for i in f:
	input_data.set_value(i,"Fare",int(mean["Fare"]))

input_data["Age"]-=mean["Age"]
input_data["Fare"]-=mean["Fare"]
input_data["Age"]/=(max(input_data["Age"])-min(input_data["Age"]))
input_data["Fare"]/=(max(input_data["Fare"])-min(input_data["Fare"]))

#print input_data["Fare"].describe() 

predict=pd.DataFrame(model2.predict(input_data))

predict=predict.convert_objects(convert_numeric=True)

predict=np.array(predict).reshape((418,1))

print predict

np.savetxt('logistic.csv',predict)
