import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Load the CSV file into a pandas dataframe, while specifying the column names
data = pd.read_csv('final-dataset.csv', names=["imagename", "temperature", "precipitationIntensity", "vegetationPercent", "latitude", "longitude", "plantHealth1", "plantHealth2", "plantHealth3", "plantHealth4", "plantHealth5", "plantHealth6", "plantHealth7", "plantHealth8", "climateID", "cloudType", "landType"])

# Separate features and labels, while removing unnecessary columns
features = data.drop('longitude', axis=1).drop('imagename', axis=1).drop('climateID', axis=1)
labels = data['climateID']



# Replace the cloud type strings in the dataframe with integer ids,
# by creating a map.
cloud_map = {}

i = 0
for cloud_string in features['cloudType'].unique(): # for each cloud type found in the dataset
    cloud_map[cloud_string] = i # create a new integer ID, and add it to the map
    i += 1

# Uncomment to print the generated cloud map
print(cloud_map)

features['cloudType'] = features['cloudType'].map(cloud_map) # replace, using the map



# Replace the land type strings in the dataframe with integer ids,
# by creating a map.
land_map = {}

i = 0
for land_string in features['landType'].unique(): # for each land type found in the dataset
    land_map[land_string] = i # create a new integer ID, and add it to the map
    i += 1

# Uncomment to print the generated land map
print(land_map)

features['landType'] = features['landType'].map(land_map) # replace, using the map



# Remove specific colums from dataframe
# These could be used in certain scenarios (such as when latitude doesn't cause overfitting or temperature simulates a thermal camera). 
# Comment / uncomment if you want to keep / remove some of them.
# features = features.drop('latitude', axis=1)                    # remove latitude from model input
# features = features.drop('temperature', axis=1)                 # remove temperature from model input
features = features.drop('precipitationIntensity', axis=1)      # remove precipitationIntensity from model input
# features = features.drop('cloudType', axis=1)                 # remove cloudType from model input
# features = features.drop('landType', axis=1)                 # remove landType from model input

# Print the final features dataframe
print(features)


# Replace the string Climate ID labels,
# by creating a map.
label_map = {}

i = 0
for label_string in labels.unique(): # for each label found in the dataset
    label_map[label_string] = i # create a new integer ID, and add it to the map
    i += 1

print(label_map) # show the generated label map

labels = labels.map(label_map) # use the map to replace the labels

# Join the features and labels,
# then use the train_test_split() function from sklearn to split the data.
# After that, split the features and labels again. 

final_dataset_merged = features
final_dataset_merged['climateID'] = labels
train_dataset, test_dataset = train_test_split(final_dataset_merged, test_size=0.1, random_state=1337)

train_features = train_dataset.drop('climateID', axis=1)
train_labels = train_dataset['climateID']

test_features = test_dataset.drop('climateID', axis=1)
test_labels = test_dataset['climateID']


# Normalize the train & test features
normalized_features = train_features#(train_features - train_features.mean()) / train_features.std()
normalized_test_features = test_features#(test_features - test_features.mean()) / test_features.std()

# Convert the preprocessed data to numpy arrays
X = normalized_features.values
y = train_labels.values

# Create a TensorFlow Dataset from the numpy arrays
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Shuffle and batch the dataset
batch_size = 32
dataset = dataset.shuffle(len(X)).batch(batch_size)

# Define the Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dropout(0.15),
    #tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(dataset, epochs=450)

# Evaluate the model
# Convert the preprocessed data to numpy arrays
X = normalized_test_features.values
y = test_labels.values

# Create a TensorFlow Dataset from the numpy arrays
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Shuffle and batch the dataset
batch_size = 32
dataset = dataset.shuffle(len(X)).batch(batch_size)

print("\n\nTest:")

test_loss, test_acc = model.evaluate(dataset)
print('Test accuracy:', test_acc)

# Show the model summary
model.summary()


model.save("climate-model")

# Convert the model .
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('climate-model.tflite', 'wb') as f:
  f.write(tflite_model)

# Plot the model validation accuracy with respect to epochs.
# This plot may be used to avoid overtraining, and find the best epochs value.

import matplotlib.pyplot as plt

# Get accuracy & loss values from the history
accuracy = history.history['accuracy']
loss = history.history['loss']

# Plot the accuracy values
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'b', label='Validation accuracy')
plt.plot(epochs, loss, 'r', label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


