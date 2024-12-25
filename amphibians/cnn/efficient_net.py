from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.applications import EfficientNetB0, EfficientNetB1
from keras import backend
from dataset_generation_amphibia import train_generator, validation_generator
import matplotlib.pyplot as plt


def train_model(learning_rate, optimizer, epochs, preset_name=str()):
    base_model = EfficientNetB1(include_top=False, weights='imagenet', input_shape=(240, 240, 3))
    base_model.trainable = True
    
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='softmax'))

    model.summary()
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
    )
    
    train_accuracy_history = history.history['accuracy']
    val_accuracy_history = history.history['val_accuracy']
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']
    model.save(f'amphibia_effb1_{preset_name}.h5')
    
    return train_accuracy_history, val_accuracy_history, train_loss_history, val_loss_history

results = []

parameter_presets = {
    'adam_notop': (0.0001, 'adam', 50),
    #'sgd': (2, 0.005, 'sgd', 50, True),
    #'Preset3': (3, 0.0001, 'adam', 50, True),
}

for preset_name, parameters in parameter_presets.items():
    backend.clear_session()
    train_accuracy_history , val_accuracy_history, train_loss_history , val_loss_history = train_model(*parameters, preset_name)
    results.append((val_accuracy_history[-1], preset_name))
    
    plt.plot(val_accuracy_history, label=f'{preset_name} - CL({parameters[0]}) DL({parameters[1]}) LR({parameters[2]}) {parameters[3]}')
    
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    # First subplot for accuracy
    axs[0].plot(train_accuracy_history, label='Train Accuracy')
    axs[0].plot(val_accuracy_history, label='Validation Accuracy')
    axs[0].set_title(f'{preset_name} - CL({parameters[0]}) DL({parameters[1]}) LR({parameters[2]}) {parameters[3]}')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    
    # Second subplot for loss
    axs[1].plot(train_loss_history, label='Train Loss')
    axs[1].plot(val_loss_history, label='Validation Loss')
    axs[1].set_title(f'{preset_name} - CL({parameters[0]}) DL({parameters[1]}) LR({parameters[2]}) {parameters[3]}')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'amphibia_effb1_{preset_name}.png')
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

