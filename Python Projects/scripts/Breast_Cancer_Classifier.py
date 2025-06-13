#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Introduction
#The purpose of this project is to apply machine learning models in order to predict whether a tumor is cancerous (malignant) or non-cancerous (benign).
#The dataset can be found here: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
#Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.


# In[2]:


###Import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from xgboost import XGBClassifier

random.seed(1)

print("All core libraries loaded successfully!")


# In[3]:


####Get the data set
###This dataset includes several measurements of tumors, along with their diagnsis: Malignant (cancer) vs benign (non-cancerous).
filename = "../data/breast_cancer.csv"
data = pd.read_csv(filename)
print("File read successfully")


# In[4]:


###Uncomment and run to preview first 5 data entries
#data.head()


# In[5]:


###Organizing the data into X and Y components, along with splitting into training, validation, and testing sets.
#Check for NaN and NULL data points. Uncomment the line below to see number of empty data slots in each column.
#print(data.isna().sum())

#Need to remove "id" and "Unnamed: 32" columns. Also remove diagnosis to get x variable.
X_total = data.drop(['id','diagnosis','Unnamed: 32'], axis=1)

#y variable will be "diagnosis". Also must convert to numeric value
#M = 1, B = 0
Y_total = data["diagnosis"]
Y_total = Y_total.map({'M': 1, 'B': 0})

#Seperating X and Y value into final train,valid,test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X_total, Y_total, test_size=0.3, random_state=1)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=1)


# In[6]:


###Checking accuracy using cross validation to ensure good fit
my_pipeline = Pipeline(steps=[('model', XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4))])
scores = -1 * cross_val_score(my_pipeline, X_total, Y_total, cv=5, scoring='accuracy')
print("Accuracy ", -1*scores.mean())


# In[7]:


###Training model on 70%, validation on 15%, testing on 15%. Then creating predictions on the test cases.
###XGBClassifier was chosen for its powerful gradient boosting algorithm with high performance.
my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4,early_stopping_rounds=5)
my_model.fit(X_train,Y_train,eval_set=[(X_valid,Y_valid)],verbose=False)
prediction = my_model.predict(X_test)


# In[8]:


###Final analysis of predictions versus actuals using classification matrix
###Sensitivity measures the ability of the model to correctly identify positive cases
###Specificity measures the ability of the model to correctly identify negative cases
cm = confusion_matrix(Y_test,prediction)
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

#Calculate and output sensitivity and specificity
print("Sensitivity: ",TP/(TP+FN)*100,"%")
print("Specificity: ",TN/(TN+FP)*100,"%")

#Calculate and output accuracy
print("Accuracy: ",accuracy_score(Y_test,prediction)*100,"%")


# In[9]:


###Setup plot format
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# In[10]:


###Visualizations of the data
#Countplot to see distribution of malignant and benign
sns.countplot(x=Y_total)
plt.xticks([0, 1], ["Benign", "Malignant"])
plt.title("Class Distribution: Benign vs Malignant")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#heatmap to show relationship between features and diagnosis
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

#ROC Curve
y_probs = my_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()


# In[11]:


###Conclusion
#The model successfully predicts whether the tumor is cancerous or not with a high accuracy of 96.5%/
#The sensitivity is 93.3% and the specificity is 98.2%.
#While the model is not perfect, it would be difficult to get a more accurate prediction without overfitting.

