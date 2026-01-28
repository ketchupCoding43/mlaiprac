import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

time = np.arange(0, 100, 0.1)
data = np.sin(time) + np.random.normal(0, 0.1, len(time))
plt.plot(time, data)
plt.title("Synthetic Time Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("time_series_data.png", bbox_inches="tight")
#plt.show()
plt.clf()

df = pd.DataFrame(data, columns=["value"])
scaler = MinMaxScaler(feature_range=(0, 1))
df["value"] = scaler.fit_transform(df[["value"]])

def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

sequence_length = 50
data_values = df["value"].values
X, y = create_sequences(data_values, sequence_length)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential([
    LSTM(50, activation="relu", input_shape=(sequence_length, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("training_validation_loss.png", bbox_inches="tight")
#plt.show()
plt.clf()

