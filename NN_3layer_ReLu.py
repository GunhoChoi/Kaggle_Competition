import csv as csv
import numpy as np
import pandas as pd

def preprocess(x):
	#read file and save as pandas dataframe
	csv_file_object = csv.reader(open(x, 'rb'))      
	header = csv_file_object.next()                            
	data=[]                            

	for row in csv_file_object:                 
	    data.append(row)                      

	data=pd.DataFrame(data,columns=header)

	#print(data.loc[data['Fare']>=0]["Survived"])
	#print(header)
	size=len(data)

	#preprocessing data
	actual_output=data["Survived"]
	input_data=data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]#

	#pclass
	p3=input_data.loc[input_data["Pclass"]==3].index
	p2=input_data.loc[input_data["Pclass"]==2].index
	for i in range(size):
		if i in p3:
			input_layer.set_value(i,"Pclass",-1)
		elif i in p2:
			input_layer.set_value(i,"Pclass",0)


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
	#input_data=input_data.convert_objects(convert_numeric=True)
	input_data=input_data.apply(pd.to_numeric)
	actual_output=actual_output.apply(pd.to_numeric)

	#age
	mean=input_data.mean()
	zero_age=input_data.loc[input_data["Age"].isnull()].index
	for i in zero_age:
		input_data.set_value(i,"Age",int(mean["Age"]))

	input_data["Age"]-=mean["Age"]
	input_data["Fare"]-=mean["Fare"]
	input_data["Age"]/=((max(input_data["Age"])-min(input_data["Age"])))/10
	input_data["Fare"]/=((max(input_data["Fare"])-min(input_data["Fare"])))/10

	preprocessed_actual_output=np.zeros((891,2))
	for i in range(size):
		preprocessed_actual_output[i,actual_output[i]]=1
	#print(actual_output,preprocessed_actual_output)

	return input_data,preprocessed_actual_output

def preprocess_test(x):
	#read file and save as pandas dataframe
	csv_file_object = csv.reader(open(x, 'rb'))      
	header = csv_file_object.next()                            
	data=[]                            

	for row in csv_file_object:                 
	    data.append(row)                      

	data=pd.DataFrame(data,columns=header)

	size=len(data)

	#preprocessing data
	input_data=data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]#


	#pclass
	p3=input_data.loc[input_data["Pclass"]==3].index
	p2=input_data.loc[input_data["Pclass"]==2].index
	for i in range(size):
		if i in p3:
			input_layer.set_value(i,"Pclass",-1)
		elif i in p2:
			input_layer.set_value(i,"Pclass",0)

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

	input_data=input_data.apply(pd.to_numeric)
	
	#age
	mean=input_data.mean()
	zero_age=input_data.loc[input_data["Age"].isnull()].index
	for i in zero_age:
		input_data.set_value(i,"Age",int(mean["Age"]))

	input_data["Age"]-=mean["Age"]
	input_data["Fare"]-=mean["Fare"]
	input_data["Age"]/=((max(input_data["Age"])-min(input_data["Age"])))/10
	input_data["Fare"]/=((max(input_data["Fare"])-min(input_data["Fare"])))/10

	return np.array(input_data)

class NN:
	num_input=7
	num_output=2
	l1_hidden_node=4
	learning_rate=0.0002
	reg=0.01

def relu(x):
	return np.maximum(0,x)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
	return x*(1-x)

def relu_derivative(x,input_layer):
	for i in range(len(x)):
		for j in range(NN.l1_hidden_node):
			if input_layer[i,j] <= 0:
				x[i,j]=0
	return x

def softmax(x):
	exp_scores=np.exp(x)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # axis=1 & keep dimension !!
	return probs

def predict(model, x):
    size=len(x)
    result=[]
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    # Forward propagation
    z1 = x.dot(w1) + b1
    a1 = relu(z1)
    z2 = a1.dot(w2) + b2
    s=softmax(z2)
    for i in range(size):
    	if s[i,0]>=s[i,1]:
    		result.append(0)
    	else:
    		result.append(1)
    result=np.array(result)
    result=np.reshape(result,(size,1))	
    return result

def NeuralNet_1_layer(input_data,output_data,hidden_nodes,iteration=30000,print_loss=False):
	size=len(input_data)
	w1=np.random.randn(NN.num_input,hidden_nodes)
	b1=np.ones((1,hidden_nodes))
	w2=np.random.randn(hidden_nodes,NN.num_output)
	b2=np.ones((1,NN.num_output))

	for i in range(iteration):
		
		# forward propagation
		layer0=input_data

		layer1_in=np.dot(input_data,w1)+b1
		layer1_out=relu(layer1_in)

		layer2_in=np.dot(layer1_out,w2)+b2
		layer2_out=softmax(layer2_in)

		# error calculation
		if i%100==0:
			cross_entropy_error=-np.log(np.sum(layer2_out*output_data,axis=1))
			data_loss= np.sum(cross_entropy_error)/size
			reg_loss=0.5*NN.reg*(np.sum(w1*w1)+np.sum(w2*w2))
			loss=data_loss+reg_loss
			print i, data_loss

		# back propagation
		error=(layer2_out-output_data)
				
		w2_update=np.dot(layer1_out.T,error)
		b2_update=np.sum(error,axis=0,keepdims=True)
		
		error_layer1=np.dot(error,w2.T)
		error_layer1=relu_derivative(error_layer1,layer1_in)

		w1_update=np.dot(layer0.T,error_layer1)
		b1_update=np.sum(error_layer1,axis=0,keepdims=True)

		w2_reg=1/NN.l1_hidden_node*NN.reg*np.sum(w2*w2)
		w1_reg=1/NN.num_input*NN.reg*np.sum(w1*w1)
		
		w2-=NN.learning_rate*(w2_update+w2_reg)
		b2-=NN.learning_rate*b2_update
		w1-=NN.learning_rate*(w1_update+w1_reg)
		b1-=NN.learning_rate*b1_update


	model = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
	
	return model

input_data,output_data=preprocess('train.csv')
#print(output_data)
model=NeuralNet_1_layer(input_data,output_data,NN.l1_hidden_node)

a=preprocess_test('test.csv')

b=predict(model,a)

np.savetxt('relu_3layer_NeuralNet.csv',b)