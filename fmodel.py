import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Read data in from file
with open("odd.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "hd": [float(cell) for cell in row[:2]],
            "a": [float(row[2])]
        })



# Separate data into training and testing groups
evidence = [row["hd"] for row in data]
labels = [row["a"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4)

print(X_training)
print(y_training)


# Create a neural network
model = tf.keras.models.Sequential()

# Add a hidden layer with 8 units, with ReLU activation
model.add(tf.keras.layers.Dense(3, input_shape=(2,)))
#model.add(tf.keras.layers.Dropout(0.4))

# Add output layer with 1 unit, with sigmoid activation
model.add(tf.keras.layers.Dense(1))

# Train neural network
model.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss="mean_squared_error",
    metrics=["accuracy"]
    )

history= model.fit(X_training, y_training, epochs=100)

# Evaluate how well model performs
model.evaluate(X_testing, y_testing, verbose=2)

print(model.predict([2.45, 3.35, 2.65]))

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
plt.show()


print(model.predict([2.95]))
