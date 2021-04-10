from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
#Importing dataset as well as classifier

iris = datasets.load_iris()
#print(iris.DESCR)

#Defining features and labels
features = iris.data
labels = iris.target
#print(features[0],labels[0])

#Training our model
clf = KNeighborsClassifier()
#Fitting our model
clf.fit(features,labels)
#Predicting
pred = clf.predict([[0,1,1,9]])

print(pred)



