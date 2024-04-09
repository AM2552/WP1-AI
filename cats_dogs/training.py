from simple_cnn import SimpleCNN
from dataset_generation import train_generator, validation_generator

# Create an instance of the model
model = SimpleCNN()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.build((None, 256, 256, 3))  # `build` is called to define the input shape so the model can be summarized
model.summary()

history = model.fit(
      train_generator,
      steps_per_epoch=100,  # Depends on your dataset size and batch size
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50)  # Depends on your validation dataset size and batch size


model.save('cats_vs_dogs_model.h5')