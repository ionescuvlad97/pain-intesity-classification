# Standard Libraries
from pathlib import Path
from joblib import dump, load
import time
import pickle

# Third Party Libraries
import numpy as np

# Custom Libraries
from utils import train_model
from utils import load_dataset


def main():
    training_type = 'pca95'
    X_train, y_train, X_test, y_test = load_dataset(train_type=training_type)
    print('Raw shapes')
    print('Training images size: {}'.format(X_train.shape))
    print('Training labels size: {}'.format(y_train.shape))
    print('Testing images size: {}'.format(X_test.shape))
    print('Testing labels size: {}'.format(y_test.shape))

    X_test_descriptor_len = X_test.shape[1]
    X_train = X_train[:, :X_test_descriptor_len]

    print('Shapes after resizing:')
    print('Training images size: {}'.format(X_train.shape))
    print('Training labels size: {}'.format(y_train.shape))
    print('Testing images size: {}'.format(X_test.shape))
    print('Testing labels size: {}'.format(y_test.shape))

    neurons_number = [10, 2500, 5000]
    learning_rates = [0.01, 0.1, 10]
    model_version = 1

    for neurons in neurons_number:
        for index, learning_rate in enumerate(learning_rates):
            train_model_start_time = time.time()
            print("Number of neurons: {}".format(neurons))
            print("Learning rate: {}".format(learning_rate))
            model = train_model(X_train, y_train, neurons, learning_rate)
            Path("models").mkdir(parents=True, exist_ok=True)
            dump(model, 'models/model_n{}_l{}_v{}.joblib'.format(
                neurons,
                index,
                model_version))
            train_model_stop_time = time.time() - train_model_start_time
            print("Model train time: {}s".format(train_model_stop_time))
            print("Accuracy: {}".format(model.score(X_test, y_test)))
            print()


if __name__ == '__main__':
    main()
