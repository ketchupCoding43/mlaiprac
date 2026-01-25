import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

os.makedirs("graphs", exist_ok=True)

df = pd.read_csv("knn_dataset.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

correct_idx = np.where(y_test == y_pred)[0]
wrong_idx = np.where(y_test != y_pred)[0]

print("\nCorrect Predictions:")
for i in correct_idx:
    print(f"Sample {i} -> True: {y_test.iloc[i]}, Predicted: {y_pred[i]}")

print("\nWrong Predictions:")
for i in wrong_idx:
    print(f"Sample {i} -> True: {y_test.iloc[i]}, Predicted: {y_pred[i]}")


if X.shape[1] == 2:
    plt.figure(figsize=(6,5))
    plt.scatter(X_test[correct_idx,0], X_test[correct_idx,1], c='green', label='Correct', marker='o')
    plt.scatter(X_test[wrong_idx,0], X_test[wrong_idx,1], c='red', label='Wrong', marker='x')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Predictions')
    plt.legend()
    plt.savefig("graphs/knn_predictions.png", bbox_inches="tight")
    plt.show()
    plt.close()
