import pandas as pd
import numpy as np

dataframe = pd.read_csv("C:\\Users\\User\\Downloads\\spamemails\\emails.csv")
print(dataframe.head())
#print(dataframe.info())
#print(dataframe['label'].value_counts())

print(dataframe['Prediction'].value_counts())


y = dataframe['Prediction'] #target column
x = dataframe.drop(['Prediction', 'Email No.'], axis=1)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


import plotly.express as px

fig = px.imshow(cm, labels={'x': "Predicted", 'y': "Actual"}, color_continuous_scale="Reds")
fig.update_layout(title="Confusion Matrix: Spam Email Classifier")
fig.show()
