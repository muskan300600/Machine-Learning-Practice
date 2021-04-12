from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
#print (iris.keys())
X = iris.data[:,3:]
y = (iris.target==2).astype(np.int)

clf = LogisticRegression()
clf.fit(X,y)
example = clf.predict(([[2]]))

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)

plt.plot(X_new,y_prob[:,1])
plt.show()






