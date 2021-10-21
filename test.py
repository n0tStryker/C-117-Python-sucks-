from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

actual_data = ["Not Sick", "Sick", "Not Sick", "Not Sick", "Sick", "Sick", "Not Sick", "Not Sick", "Not Sick", "Not Sick", "Not Sick", "Not Sick"] 
predicted_data = ["Not Sick", "Sick", "Not Sick", "Not Sick", "Not Sick", "Sick", "Not Sick", "Sick", "Not Sick", "Not Sick", "Sick", "Not Sick"]

labels = ["Not Sick", "Sick"]
cm = confusion_matrix(actual_data, predicted_data, labels)

ax = plt.subplot()
sns.heatmap(cm,annot = True, ax = ax)

ax.set_xlabel('predicted')
ax.set_ylabel('actual')
ax.set_title('confusion matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)