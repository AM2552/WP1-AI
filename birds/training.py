from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from dataset_generation import train_generator, validation_generator

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # Adjust based on your dataset size and batch size
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50)  # Adjust based on your validation dataset size and batch size

# Save the model in TensorFlow's SavedModel format
model.save('cats_vs_dogs_model.h5')