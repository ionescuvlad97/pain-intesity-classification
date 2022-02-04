# Standard Libraries
from pathlib import Path
import pickle

# Third Party Libraries
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import color
from skimage.feature import hog
from skimage import exposure
from sklearn.neural_network import MLPClassifier
from skimage.transform import resize
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

# Custom Libraries

# Global Variables
dataset_name = "UNBC"
train_data_path = r'D:\Datasets\{}\Images\Train'.format(dataset_name)
test_data_path = r'D:\Datasets\{}\Images\Test'.format(dataset_name)
train_labels_path = r'D:\Datasets\{}\Labels\Train'.format(dataset_name)
test_labels_path = r'D:\Datasets\{}\Labels\Test'.format(dataset_name)


def show_images(images, length, width):
    for i in range(length*width):
        plt.subplot(length, width, 1+i)
        plt.axis('off')
        plt.imshow(images[i], cmap='gray')


def class_reduction(label):
    if label == 0:
        return 0
    elif label == 1 or label == 2:
        return 1
    elif label == 3 or label == 4:
        return 2
    elif label >= 5:
        return 3


def data_generator(path, b_size, image_shape):
    result = []
    for image_path in Path(path).rglob('*.png'):
        image = plt.imread(image_path)
        image_resized = resize(image, image_shape, anti_aliasing=True)
        result.append(image_resized)
        if len(result) % b_size == 0:
            yield np.array(result)
            result = []


def labels_generator(path, b_size):
    result = []
    for label_path in Path(path).rglob('*.txt'):
        with open(str(label_path), 'r') as fp:
            label = class_reduction(int(float(fp.readline().strip())))
            result.append(label)
            if len(result) % b_size == 0:
                yield np.array(result)
                result = []


def get_dataset(data_path, labels_path, image_shape):
    X, y = [], []
    for image_path, label_path in zip(Path(data_path).rglob('*.png'), Path(labels_path).rglob('*.txt')):
        image = plt.imread(image_path)
        image_resized = resize(image, image_shape, anti_aliasing=True)
        X.append(image_resized)
        with open(str(label_path), 'r') as fp:
            y.append(class_reduction(int(float(fp.readline().strip()))))
    return np.array(X), np.array(y)


def convert_rgb2gray(images):
    return np.array([color.rgb2gray(image) for image in images])


def get_hog(images):
    fd_list, hog_image_list = [], []
    for image in images:
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        fd_list.append(fd)
        hog_image_list.append(exposure.rescale_intensity(hog_image, in_range=(0, 50)))
    return np.array(fd_list), np.array(hog_image_list)


def prepare_images(batch_size, image_shape, number_of_iterations, variance, images_type='train'):
    path = train_data_path if images_type == 'train' else test_data_path
    print(f'{path=}')
    generator = data_generator(path, batch_size, image_shape)
    X = []
    count = 0
    for batch in generator:
        X_hog, _ = get_hog(convert_rgb2gray(batch))
        X.extend(X_hog)
        count += 1
        if count == number_of_iterations:
            break
    X = np.array(X)

    pca = PCA(variance)
    pca.fit_transform(X.T)

    X_hog_pca = pca.components_

    Path("data").mkdir(parents=True, exist_ok=True)

    with open("data/{}_fd_hog_pca{}.pkl".format(images_type, str(variance)[2:]), 'wb') as f:
        pickle.dump(X_hog_pca.T, f)


def prepare_labels(batch_size, number_of_iterations, labels_type='train'):
    path = train_labels_path if labels_type == 'train' else test_labels_path
    print(f'{path=}')
    generator = labels_generator(path, batch_size)
    y = []
    count = 0
    for batch in generator:
        y.extend(batch)
        count += 1
        if count == number_of_iterations:
            break
    y = np.array(y)

    Path("data").mkdir(parents=True, exist_ok=True)

    with open("data/{}_labels.pkl".format(labels_type), 'wb') as f:
        pickle.dump(y, f)


def load_dataset(train_type='pca99'):
    with open("data/train_fd_hog_{}.pkl".format(train_type), 'rb') as f:
        X_train = np.array(pickle.load(f))
    with open("data/test_fd_hog_{}.pkl".format(train_type), 'rb') as f:
        X_test = np.array(pickle.load(f))
    with open('data/train_labels.pkl', 'rb') as f:
        y_train = np.array(pickle.load(f))
    with open('data/test_labels.pkl', 'rb') as f:
        y_test = np.array(pickle.load(f))

    return X_train, y_train, X_test, y_test


# def train_model(train_data_path, train_label_path, batch_size, image_size, hidden_layer_size, learning_rate):
#     clf = MLPClassifier(
#             hidden_layer_sizes=hidden_layer_size,
#             learning_rate_init=learning_rate,
#             max_iter=10,
#             verbose=True
#         )
#     for X_train, y_train in zip(data_generator(train_data_path, batch_size, image_size),
#                                 labels_generator(train_label_path, batch_size)):
#         X_train_hog, _ = get_hog(convert_rgb2gray(X_train))
#         clf.partial_fit(X_train_hog, y_train, classes=[i for i in range(4)])
#     return clf

def train_model(X_train, y_train, hidden_layer_size, learning_rate):
    clf = MLPClassifier(
                hidden_layer_sizes=hidden_layer_size,
                learning_rate_init=learning_rate,
                verbose=True
            )
    clf.fit(X_train, y_train)
    return clf
