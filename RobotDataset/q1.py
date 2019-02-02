import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as sk_knn
import matplotlib.pyplot as plt



data=pd.read_csv('Robot1',sep=' ',header=None)


def euclidean(a,b):
	sum=0
	for count,i in enumerate(a):
		x= a[count]- b[count]
		sum= sum + x*x

	res= math.sqrt(sum)

	return res




def knn(x,y,xtest,k):
	columns=x.columns
	res=[]
	for rowtest in xtest.itertuples():
		pnt2=[i for i in rowtest]
		pnt2.pop(0)
		l=[]
		
		for row in x.itertuples():
			pnt1= [i for i in row]
			pnt1.pop(0)
			toappend=[]

			dist= euclidean(pnt1,pnt2)
			toappend.append(dist)
			toappend.append(row.Index)
			l.append(toappend)

		l.sort()    #[[dist,index],[..],[..]]
		classes= set(y)
		d={}
		for c in classes:
			d[c]=0

		for i in range(k):
			ind= l[i][1]
			label = y[ind]
			d[label] = d[label]+1

		maxv=0

		for i in d:
			if(d[i]>maxv):
				maxv=d[i]
				key=i

		res.append(key)

	return res




def accuracy(ytest,ypredict):
	c=0
	l =len(ytest)

	for count,i in enumerate(ytest):
		if(ytest[count] == ypredict[count]):
			c= c +1
	# print("count= ",c)
	# print("total= ",l)

	return c/l


# def parameters(ytest,ypredict):

# 	tp=0
# 	fp=0
# 	fn=0
# 	for count,i in enumerate(ytest):
# 		if(ypredict[count]==true and ytest[count]==true):
# 			tp= tp+1
# 		if(ypredict[count]==true and ytest[count]==false):
# 			fp = fp+1
# 		if(ypredict[count]==false and ytest[count]== true):
# 			fn = fn +1
# 	l=[tp,fp,fn]
# 	print("tp fp fn= ",l)

# 	return l


# def recall(tp,fn):
# 	den = tp+fn
# 	if(den==0):
# 		return 0.0
# 	return tp/den

# def precision(tp,fp):
# 	den= tp+fp
# 	if(den==0):
# 		return 0.0
# 	return tp/den

# def f1_score(precision,recall):
# 	pr1= 1/precision

# data2=data
data2= data[['Work_accident','promotion_last_5years', 'sales', 'salary','left']]
value_dict= prepare_dict(data2)
print("value dict= ",value_dict)
features = data2.columns
features=features.drop(label)
print("features= ",features)
# print("data2= ",data2)
x=data2.drop(columns=label)
y= data2[label]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)
l= [xTrain,yTrain]
data_train=pd.concat(l,axis=1)
# print("data train= ",data_train)
l=[xTest,yTest]
data_test = pd.concat(l,axis=1)








		










col=['none','class']

i='a'
k='a1'

for j in range(2,8):
	col.append(k)
	k=i+str(j)

col.append('id')
data.columns= col

x=data[['a1','a2','a3','a4','a5','a6']]
y=data['class']
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)
yTest= yTest.tolist()
# for i in range(5,35):
for i in range(2,50):
	ypredict=knn(xTrain,yTrain,xTest,i)

	print("accuracy for i = ",i)
	print(accuracy(yTest,ypredict))

for i in range(2,50):
	print("i= ",i)
	neigh = sk_knn(n_neighbors=i)
	neigh.fit(xTrain,yTrain)
	ypredict= neigh.predict(xTest)
	print("sklearn accuracy= ",accuracy_score(yTest,ypredict))

# print(col)

