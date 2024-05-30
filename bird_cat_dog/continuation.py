from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, SGD
from keras import backend
from bird_cat_dog.dataset_generation_bcd import train_generator, validation_generator
import matplotlib.pyplot as plt

last_trained_epoch = 200
additional_epochs = 200  # for example, to train for 100 more epochs
preset_name = 'Preset1'
model_path = f'bird_cat_dog_model_916.h5'
model = load_model(model_path)
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Adjust if needed
    epochs=last_trained_epoch + additional_epochs,
    validation_data=validation_generator,
    validation_steps=50,  # Adjust if needed
    initial_epoch=last_trained_epoch  # Set to the last epoch of the previous training
)
    
new_accuracy_history = history.history['val_accuracy']
results = []
results.append((new_accuracy_history[-1], preset_name))
accuracy_history = history.history['val_accuracy']
model.save(f'bird_classification_model_916_continued.h5')  # Updated model naming

parameter_presets = {
    'Preset1': (6, 3, 0.0001, 'adam', 300, True),
    'Preset2': (6, 3, 0.0001, 'adam', 300, True),
    #'Preset3': (6, 3, 0.0002, 'adam', 300, True),
    #'Preset4': (6, 3, 0.0001, 'sgd', 300, True)
}

plt.plot(range(last_trained_epoch, last_trained_epoch + additional_epochs), new_accuracy_history, label=f'Continued {preset_name} - CL(6) DL(3) LR(0.00009) adam')
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

plt.title('Model accuracy - Continued Training')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('bird_classification_accuracy_continued.png')
plt.show()
