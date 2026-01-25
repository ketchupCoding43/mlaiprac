import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

os.makedirs("graphs", exist_ok=True)

# Example dataset
data = {
    "Feature1": [5.1,4.9,4.7,4.6,5.0,5.4,4.6,5.0,4.4,4.9,7.0,6.4,6.9,5.5,6.5,5.7,6.3,4.9,6.6,5.2],
    "Feature2": [3.5,3.0,3.2,3.1,3.6,3.9,3.4,3.4,2.9,3.1,3.2,3.2,3.1,2.3,2.8,2.8,3.3,2.4,2.9,2.7],
    "Label":    [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Feature1", "Feature2"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Limit tree depth to avoid overfitting
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize tree
plt.figure(figsize=(10,6))
plot_tree(tree, feature_names=X.columns, class_names=["0","1"], filled=True)
plt.title("Decision Tree Classifier")
plt.savefig("graphs/decision_tree.png", bbox_inches="tight")
plt.show()
plt.close()
