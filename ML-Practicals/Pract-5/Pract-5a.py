import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

# -----------------------------
# 1. Create Dataset
# -----------------------------
data = {
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temp": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
    "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "Play": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Separate Features & Target
# -----------------------------
X = df.drop("Play", axis=1)
y = df["Play"]

# -----------------------------
# 3. Encode Categorical Data
# -----------------------------
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# Encode target (Yes=1, No=0)
y_encoded = y.map({"No": 0, "Yes": 1})

# -----------------------------
# 4. Train Naive Bayes Model
# -----------------------------
model = CategoricalNB()
model.fit(X_encoded, y_encoded)

# -----------------------------
# 5. Test Sample (FIXED PART)
# -----------------------------
test_sample = pd.DataFrame(
    [["Sunny", "Cool", "High", "Strong"]],
    columns=["Outlook", "Temp", "Humidity", "Wind"]
)

test_encoded = encoder.transform(test_sample)

# -----------------------------
# 6. Prediction
# -----------------------------
prediction = model.predict(test_encoded)
probabilities = model.predict_proba(test_encoded)

result = "Yes" if prediction[0] == 1 else "No"

print("Test Sample:", test_sample.values[0])
print("Prediction (Play Tennis?):", result)
print("Probability [No, Yes]:", probabilities[0])
