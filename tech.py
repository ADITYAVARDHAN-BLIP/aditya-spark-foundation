
              ###prediction using decision tree algorithm

#import important library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#import the dataset
df=pd.read_csv("Iris.csv")

#pre-processing
df=df.replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
x=df.iloc[:,1:-1]
y=df.iloc[:,-1]

#select and train the model
dtree=DecisionTreeClassifier()
dtree.fit(x,y)

#plot the decision tree
fig=plt.figure(figsize=(9,9))
_ = tree.plot_tree(dtree,
                   feature_names=x.columns,
                   class_names=['0','1','2'],
                   filled=True)
plt.show()

