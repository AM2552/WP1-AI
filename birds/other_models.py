from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras import backend
from dataset_generation_birds import train_generator, validation_generator
import matplotlib.pyplot as plt

def lr_schedule(epoch, lr):
    if epoch > 30:
        return float(lr * 0.01)
    elif epoch > 15:
        return float(lr * 0.1)

def train_model(dense_layers, initial_learning_rate, optimizer_name, epochs, dropout=True, preset_name=''):
    base_model = EfficientNetB1(include_top=False, weights=None, input_shape=(256, 256, 3))
    base_model.trainable = True  # Train the base model from scratch
    
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    
    neurons = 1024
    for _ in range(dense_layers):
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        neurons = int(neurons * 0.7)
        if dropout:
            model.add(Dropout(0.5))
    
    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(200, activation='softmax'))

    model.summary()
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=initial_learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=initial_learning_rate, momentum=0.9)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lr_schedule(lr=initial_learning_rate))

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[lr_scheduler]
    )
    
    train_accuracy_history = history.history['accuracy']
    val_accuracy_history = history.history['val_accuracy']
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    model.save(f'birds_eff1_{preset_name}.h5')
    
    return train_accuracy_history, val_accuracy_history, train_loss_history, val_loss_history

results = []

parameter_presets = {
    'Preset1': (2, 0.01, 'sgd', 45, True),
    #'Preset2': (3, 0.0001, 'adam', 50, True),
    #'Preset3': (3, 0.0001, 'adam', 50, True),
}

for preset_name, parameters in parameter_presets.items():
    backend.clear_session()
    train_accuracy_history , val_accuracy_history, train_loss_history , val_loss_history = train_model(*parameters, preset_name)
    results.append((val_accuracy_history[-1], preset_name))
    
    plt.plot(val_accuracy_history, label=f'{preset_name} - DL({parameters[0]}) LR({parameters[1]}) {parameters[2]}')

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    # First subplot for accuracy
    axs[0].plot(train_accuracy_history, label='Train Accuracy')
    axs[0].plot(val_accuracy_history, label='Validation Accuracy')
    axs[0].set_title(f'{preset_name} - DL({parameters[0]}) LR({parameters[1]}) {parameters[2]}')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    
    # Second subplot for loss
    axs[1].plot(train_loss_history, label='Train Loss')
    axs[1].plot(val_loss_history, label='Validation Loss')
    axs[1].set_title(f'{preset_name} - DL({parameters[0]}) LR({parameters[1]}) {parameters[2]}')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'birds_eff7_{preset_name}.png')
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')
