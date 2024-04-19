from keras import Sequential
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D
import tensorflow as tf


def getEncoder(shape):
    """Get the encoder network."""
    model = Sequential()
    model.add(Input(shape=shape))
    model.add(Conv2D(10, [3,3], activation="relu"))
    model.add(MaxPool2D([2,2]))
    model.add(Conv2D(10, [3,3], activation="relu"))
    model.add(MaxPool2D([2,2]))
    model.add(Flatten())
    model.add(Dense(20, activation="relu"))
    return model

def getTask():
    """Get the task network."""
    model = Sequential()
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation="softmax"))
    return model

def getDiscriminator():
    """Get the discriminator network."""
    model = Sequential()
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


class EarlyStoppingCustom(tf.keras.callbacks.Callback):
    def __init__(self):
        super(EarlyStoppingCustom, self).__init__()
        self.best_accuracy = 0.0
        self.best_disc_accuracy = 0.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        # Get current metrics from the logs
        current_accuracy = logs.get('accuracy')
        current_disc_accuracy = logs.get('disc_acc')
        
        # Update the best accuracy if current accuracy is higher
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.wait = 0  # Reset the wait counter if accuracy improves
        else:
            self.wait += 1
        
        # Update the best disc_accuracy if current disc_accuracy is higher
        if current_disc_accuracy >= self.best_disc_accuracy:
            self.best_disc_accuracy = current_disc_accuracy
        
        # Check the stopping conditions
        if self.best_disc_accuracy >= 0.6 and current_disc_accuracy < 0.55 and current_accuracy >= 0.95:
            print(f"\nStopping early: disc_accuracy dropped below 0.55. Current disc_accuracy: {current_disc_accuracy:.4f}")
            print(f"\nStopping early: disc_accuracy has exceded 0.60. Max disc_accuracy: {self.best_disc_accuracy:.4f}")
            print(f"\nStopping early: accuracy reached or surpassed 0.9. Current accuracy: {current_accuracy:.4f}")
            self.model.stop_training = True
            return
