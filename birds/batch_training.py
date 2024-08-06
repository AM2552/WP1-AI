from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, SpatialDropout2D
from keras.optimizers import Adam, SGD
from keras import backend
from dataset_generation_birds import train_generator, validation_generator
import matplotlib.pyplot as plt


def conv_layer(model, filters):
    model.add(Conv2D(filters, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


def train_model(conv_layers, learning_rate, optimizer, epochs, preset_name=str()):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(240, 240, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    filters = 64
    for _ in range(conv_layers - 1):
        conv_layer(model, filters)
        filters *= 2

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(200, activation='softmax'))
   
    model.summary()
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs
    )
    
    train_accuracy_history = history.history['accuracy']
    val_accuracy_history = history.history['val_accuracy']
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    model.save(f'birds_{preset_name}.h5')
    return train_accuracy_history, val_accuracy_history, train_loss_history, val_loss_history

results = []

parameter_presets = {
    'adam': (6, 0.001, 'adam', 30),
    #'sgd': (6, 0.01, 'sgd', 50),
    #'Preset3': (6, 0.001, 'adam', 50),
    #'Preset4': (6, 0.0001, 'sgd', 300)
}

for preset_name, parameters in parameter_presets.items():
    backend.clear_session()
    train_accuracy_history , val_accuracy_history, train_loss_history , val_loss_history = train_model(*parameters, preset_name)
    results.append((val_accuracy_history[-1], preset_name))
    
    plt.plot(val_accuracy_history, label=f'{preset_name} - CL({parameters[0]}) LR({parameters[1]})')

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    # subplot for accuracy
    axs[0].plot(train_accuracy_history, label='Train Accuracy')
    axs[0].plot(val_accuracy_history, label='Validation Accuracy')
    axs[0].set_title(f'{preset_name} - CL({parameters[0]}) LR({parameters[1]})')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    
    # subplot for loss
    axs[1].plot(train_loss_history, label='Train Loss')
    axs[1].plot(val_loss_history, label='Validation Loss')
    axs[1].set_title(f'{preset_name} - CL({parameters[0]}) LR({parameters[1]})')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'birds_{preset_name}.png')
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')
