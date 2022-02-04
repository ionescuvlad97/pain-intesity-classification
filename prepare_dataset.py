# Standard Libraries
import pickle

# Third Party Libraries
import numpy as np

# Custom Libraries
from utils import prepare_images
from utils import prepare_labels


def main():
    image_shape = (300, 200)
    batch_size = 256
    variance = 0.99
    number_of_iterations_training = 140
    number_of_iterations_testing = 40

    prepare_images(batch_size, image_shape, number_of_iterations_training, variance, images_type='train')
    prepare_images(batch_size, image_shape, number_of_iterations_testing, variance, images_type='test')

    prepare_labels(batch_size, number_of_iterations_training, labels_type='train')
    prepare_labels(batch_size, number_of_iterations_testing, labels_type='test')


def test():
    with open("data/train_fd_hog_pca99.pkl", 'rb') as f:
        X_train = np.array(pickle.load(f))
    print(X_train.shape)
    with open("data/test_fd_hog_pca99.pkl", 'rb') as f:
        X_test = np.array(pickle.load(f))
    print(X_test.shape)
    with open('data/train_labels.pkl', 'rb') as f:
        y_train = np.array(pickle.load(f))
    print(y_train.shape)
    with open('data/test_labels.pkl', 'rb') as f:
        y_test = np.array(pickle.load(f))
    print(y_test.shape)


if __name__ == '__main__':
    # main()
    test()
