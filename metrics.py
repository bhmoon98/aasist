import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt

# Load the data
data = []
with open('/home/moonbo/aasist/exp_result/DF_AASIST_1mdfdc_ep50_bs64/metrics/dev_score_3.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        data.append((parts[1], float(parts[3])))

# Extract labels and scores
labels = np.array([int(label) for label, score in data])
scores = np.array([score for label, score in data])

# Define the threshold
threshold = 0

# Calculate predictions based on the threshold
predictions = (scores >= threshold).astype(int)

# Calculate confusion matrix
conf_matrix = confusion_matrix(labels, predictions)

# Calculate precision, recall, and AUC for positive class (label 1)
precision_pos = precision_score(labels, predictions)
recall_pos = recall_score(labels, predictions)
auc = roc_auc_score(labels, scores)

# Extracting each value from confusion matrix
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

# Print results
print("Confusion Matrix:")
print(conf_matrix)
print(f"True Negative (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")
print(f"True Positive (TP): {TP}")
print(f"Precision (Positive class): {precision_pos:.4f}")
print(f"Recall (Positive class): {recall_pos:.4f}")
print(f"AUC Score: {auc:.4f}")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(labels, scores)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig('/home/moonbo/aasist/exp_result/DF_AASIST_1mdfdc_ep50_bs64/metrics/roc_curve_0fake.png')

