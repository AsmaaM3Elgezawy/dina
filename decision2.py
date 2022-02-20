# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Function importing Dataset
#def importdata():
col_name =['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
dataset= pd.read_csv("heart.csv")
print("Data:-")
print(dataset.head())# read 5 rows	
	
	# Printing the dataswet shape
#print ("Dataset Length: ", len(dataset))
print ("Dataset Shape: ", dataset.shape)
	
	# Printing the dataset obseravtions
print ("Dataset: ",dataset.head())
#	return dataset

# Function to split the dataset
#def splitdataset(balance_data):

	# Separating the target variable
features=dataset.iloc[:,:-1]
label=dataset.iloc[:,13]

	# Splitting the dataset into train and test
	features_train, features_test ,label_train ,label_test=train_test_split(features,label,test_size=0.3, random_state=1)
     print("training data ")
     print( features_train )
     print("   ")
     print(label_train)
     print("   ")
     print( "testing data")
     print( features_test )
     print("   ")
     print(label_test)
	
	#return features, label,features_train, features_test ,label_train ,label_test
	
# Function to perform training with giniIndex.
#def train_using_gini(X_train, X_test, y_train):

	# Creating the classifier object
	clf_gini = DecisionTreeClassifier(criterion = "gini",
			random_state = 100,max_depth=3, min_samples_leaf=5)

	# Performing training
	clf_gini.fit(X_train, y_train)
	return clf_gini
	
# Function to perform training with entropy.
#def tarin_using_entropy(X_train, X_test, y_train):

	# Decision tree with entropy
	clf_entropy = DecisionTreeClassifier(
			criterion = "entropy", random_state = 100,
			max_depth = 3, min_samples_leaf = 5)

	# Performing training
	clf_entropy.fit(X_train, y_train)
	return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):

	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	print(y_pred)
	return y_pred
	
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
	
	print("Confusion Matrix: ",
		confusion_matrix(y_test, y_pred))
	
	print ("Accuracy : ",
	accuracy_score(y_test,y_pred)*100)
	
	print("Report : ",
	classification_report(y_test, y_pred))
    #pred=clf.predict(np.array([[52,1,0,125,212,0,1,168,0,1,2,2,3]]))
    #print("****",pred)


   # plt.figure(figsize=(25,10))
   # a=plot_tree(clf, feature_names=col_name.remove("target"),class_names=['0','1'],filled=True,rounded=True,fontsize=14)



# Driver code
