import pandas as pd
import numpy as np
    
dataframe = pd.read_csv("C:\\Users\\Nandi\\Downloads\\spamemails\\emails.csv")
print(dataframe.head())
#print(dataframe.info())
    
#print(dataframe['label'].value_counts())


print(dataframe['Prediction'].value_counts())


y = dataframe['Prediction'] #target column
x = dataframe.drop(['Prediction', 'Email No.'], axis=1) #drop the prediction column as it is being used as the target column


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# x_train is the data taken for training the model
# x_test is the data used for testing the model
# the same applies to both y_train and y_test

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler() #used for standardizing given data to ensure every measure is the same
x_train = scaler.fit_transform(x_train) # scaling is applied based on data obtained from the training
x_test = scaler.transform(x_test) # takes parameters from training data to apply the same to test data


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression() # used for binary classification
model.fit(x_train, y_train) # establishes relationship b/w the inputs and the labels
y_pred = model.predict(x_test) # predicts class labels
accuracy = accuracy_score(y_test, y_pred) # compares the testing data with the predictions to determine accuracy
print(accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred) # compare actual vs predicted labels
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
# annot = count values
# fmt = denoted by integers
# cmap = color for the heatmap display
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

import plotly.express as px

fig = px.imshow(cm, labels={'x': "Predicted", 'y': "Actual"}, color_continuous_scale="Reds")
# labels, x = x axis labels
# labels, y = y axis labels
fig.update_layout(title="Confusion Matrix: Spam Email Classifier")
fig.show()
