from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, SGD
from keras import backend
from dataset_generation_bcd import train_generator, validation_generator
import matplotlib.pyplot as plt

last_trained_epoch = 50
additional_epochs = 50  # for example, to train for 100 more epochs
preset_name = 'Preset2'
model_path = f'bird_cat_dog_model_{preset_name}_eff.h5'
model = load_model(model_path)
history = model.fit(
    train_generator,
    epochs=last_trained_epoch + additional_epochs,
    validation_data=validation_generator,
    initial_epoch=last_trained_epoch  # Set to the last epoch of the previous training
)
    
new_accuracy_history = history.history['val_accuracy']
results = []
results.append((new_accuracy_history[-1], preset_name))
accuracy_history = history.history['val_accuracy']
model.save(f'bird_classification_model_{preset_name}_continued.h5')  # Updated model naming

parameter_presets = {
    'Preset1': (6, 3, 0.0001, 'adam', 300, True),
    'Preset2': (6, 3, 0.0001, 'adam', 300, True),
    #'Preset3': (6, 3, 0.0002, 'adam', 300, True),
    #'Preset4': (6, 3, 0.0001, 'sgd', 300, True)
}

plt.plot(range(last_trained_epoch, last_trained_epoch + additional_epochs), new_accuracy_history, label=f'Continued {preset_name} - eff DL(3) LR(0.0001) adam')
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

plt.title('Model accuracy - Continued Training')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('bcd_continued.png')
plt.show()
