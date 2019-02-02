import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as sk_knn
import matplotlib.pyplot as plt



data=pd.read_csv('Iris.csv')


def euclidean(a,b):
	sum=0
	for count,i in enumerate(a):
		x= a[count]- b[count]
		sum= sum + x*x

	res= math.sqrt(sum)

	return res

def manhattan(a,b):
	sum=0

	for count,i in enumerate(a):
		sum = sum + (a[count]- b[count])

	return sum


def chebyshev(a,b):

	l=[]

	for count,i in enumerate(a):
		l.append(a[count]-b[count])

	return max(l)


def knn(x,y,xtest,k,distance):
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
			if(distance=="euclidean"):
				dist= euclidean(pnt1,pnt2)
			elif(distance=="manhattan"):
				dist= manhattan(pnt1,pnt2)
			elif(distance=="chebyshev"):
				dist=chebyshev(pnt1,pnt2)
			else:
				print("INVALID DISTANCE MEASURE")
				return
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










		










col= ["sepal_length","sepal_width","petal_length","petal_width","class"]

data.columns= col

x=data[["sepal_length","sepal_width","petal_length","petal_width"]]
y=data['class']
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)
yTest= yTest.tolist()
# for i in range(5,35):
xplt=[]
yplt=[]
xplt_man=[]
yplt_man=[]
xplt_cheb=[]
yplt_cheb=[]
for i in range(2,50):
	ypredict=knn(xTrain,yTrain,xTest,i,"euclidean")

	print("accuracy for i = ",i)
	acc=accuracy(yTest,ypredict)
	print("acc= ",acc)

	xplt.append(i)
	yplt.append(acc*100)
	

	ypredict_man=knn(xTrain,yTrain,xTest,i,"manhattan")
	print("accuracy for i = ",i)
	acc=accuracy(yTest,ypredict_man)
	print("acc= ",acc)

	xplt_man.append(i)
	yplt_man.append(acc*100)
	
	ypredict_cheb=knn(xTrain,yTrain,xTest,i,"chebyshev")
	print("accuracy for i = ",i)
	acc=accuracy(yTest,ypredict_cheb)
	print("acc= ",acc)

	xplt_cheb.append(i)
	yplt_cheb.append(acc*100)

xplt_sk=[]
yplt_sk=[]
for i in range(2,50):
	print("i= ",i)
	neigh = sk_knn(n_neighbors=i)
	neigh.fit(xTrain,yTrain)
	ypredict= neigh.predict(xTest)
	acc=accuracy_score(yTest,ypredict)
	print("sklearn accuracy= ",acc)
	xplt_sk.append(i)
	yplt_sk.append(acc*100)


print("x and y for sklearn",xplt_sk)
print("y = ",yplt_sk)
plt.plot(xplt,yplt,c='red',label='euclidean')
plt.plot(xplt_man,yplt_man,c='green',label='manhattan')
plt.plot(xplt_cheb,yplt_cheb,c='blue',label='chebyshev')
plt.plot(xplt_sk,yplt_sk,c='yellow',label='sklearn')
plt.legend()
plt.xlabel('K')
plt.ylabel('Accuracy in percantage')
plt.show()

# for i in range(2,50):
# 	print("i= ",i)
# 	neigh = sk_knn(n_neighbors=i)
# 	neigh.fit(xTrain,yTrain)
# 	ypredict= neigh.predict(xTest)
# 	print("sklearn accuracy= ",accuracy_score(yTest,ypredict))

# print(col)

