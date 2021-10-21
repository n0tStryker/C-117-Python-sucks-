from sklearn.model_selection import train_test_split
age = df["age"]
heart_attack = df["target"]
age_train,age_test, heart_attack_train, heart_attack_test = train_test_split(age,heart_attack,test_size=0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.reshape(age_train.ravel(), (len(age_train), 1))
Y = np.reshape(heart_attack_train.ravel(), (len(heart_attack_train), 1))
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, Y)

X_test = np.reshape(age_test.ravel(), (len(age_test), 1)) 
Y_test = np.reshape(heart_attack_test.ravel(), (len(heart_attack_test), 1))
heart_attack_prediction = classifier.predict(X_test) 
predicted_values = [] 
for i in heart_attack_prediction: 
  if i == 0: 
    predicted_values.append("No") 
  else: 
    predicted_values.append("Yes") 
    
actual_values = [] 

for i in Y_test.ravel(): 
  if i == 0: 
    actual_values.append("No") 
  else: 
    actual_values.append("Yes")

labels = ["Yes", "No"]

cm = confusion_matrix(actual_values, predicted_values, labels)

ax = plt.subplot()
sns.heatmap(cm,annot = True, ax = ax)

ax.set_xlabel('predicted')
ax.set_ylabel('actual')
ax.set_title('confusion matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)