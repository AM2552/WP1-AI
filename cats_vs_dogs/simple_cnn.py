import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')
        self.pool1 = MaxPooling2D(2, 2)
        self.conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.pool2 = MaxPooling2D(2, 2)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


