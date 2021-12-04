# Pain Intensity Classification
Neural Network that generates color images from gray scale images using a Convolutional Autoencoder Network 
and GAN Architecture.
## Table of Contents

- [About](#about)
- [Database](#database)

## About <a name="about"></a>
The goal of this project is to implement a *Pain Intensity Classifier* using **Scikit-image** for image processing and
**Scikit-Learn** for machine learning.</br> </br>
A *Histogram of Oriented Gradients* (***HoG***) approach is used to extract the form and texture features. 
Firstly, the images are resized and converted into gray space to reduce the dimensionality of the feature vectors. 
HoG is applied to the reshaped images. The features extracted after applying HoG are used for classification.</br> </br>
For the classification part, one classifier is used: ***MLP***
## Database <a name="database"></a>
The database used is UNBC-McMAster Shoulder Pain Expression Archive Database.

## Built Using
- [Scikit-image](https://scikit-image.org/) - *Image Processing*
- [Scikit-learn](https://scikit-learn.org/stable/) - *Machine Learning*
- [Matplotlib](https://matplotlib.org/) - *Data Visualization*