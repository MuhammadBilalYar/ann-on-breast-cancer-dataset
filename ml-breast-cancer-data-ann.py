
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing data
data = pd.read_csv('data.csv')
del data['Unnamed: 32']

#Feature and Label selection
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set (30 - 70)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# **Now that we have prepared data, we will import Keras and its packages.**

# In[]:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
classifier = Sequential()

#===================================================================================================
# input_dim - number of columns of the dataset 
# 
# output_dim - number of outputs to be fed to the next layer, if any
# 
# activation - activation function which is ReLU in this case
# 
# init - the way in which weights should be provided to an ANN
#===================================================================================================

#----------------------------------------------------------------
#   INPUT LAYER
# input_dim: 30, output_dim: 16, init: uniform, activation:ReLu 
#
#   HIDDEN LAYER
# output_dim: 16, init: uniform, activation:ReLu 
#
#   OUTPUT LAYER
# output_dim:1, init: uniform, activation: sigmoid

#----------------------------------------------------------------
# In[]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))


# Adding the second hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))


# Adding the output layer (output_dim is 1 as we want only 1 output from the final layer.)
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


# Optimizer is chosen as adam for gradient descent.
# 
# Binary_crossentropy is the loss function used. 
# 
# Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. 

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Batch size defines number of samples that going to be propagated through the network.
# 
# An Epoch is a complete pass through all the training data.

#---------------------------------------------------------
#   Epoch: 50
#---------------------------------------------------------

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=50)

# In[]:

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print("[Epoch:50] Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/175)*100))

sns.heatmap(cm,annot=True)
plt.savefig('epoch50.png')

#---------------------------------------------------------
#   Epoch: 100
#---------------------------------------------------------

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=100)

# In[]:

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print("[Epoch:100] Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/175)*100))

sns.heatmap(cm,annot=True)
plt.savefig('epoch100.png')

#---------------------------------------------------------
#   Epoch: 150
#---------------------------------------------------------

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=150)

# In[]:

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print("[Epoch:150] Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/175)*100))

sns.heatmap(cm,annot=True)
plt.savefig('epoch150.png')


from ann_visualizer.visualize import ann_viz;

ann_viz(classifier, title="Artificial Neural Network (ANN) implementation on Breast Cancer Wisconsin Data Set")