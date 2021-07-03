# Deep-Learning
This repository consists of .ipynb files, in the scope of the second semester "Deep Learning" exersice in Master in AI in NSCR Demokritos.

Overview of the files:
* [Cameras.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Cameras.ipynb) - A Jupyter notebook file where we visualize cameras in Greece, which we use to predict if there is a fire or not.
* [Custom_CNN.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Custom_CNN.ipynb) - A Jupyter notebook file where we created a Convolutional Neural Network from scratch, which is used to predict in images if there is a fire or not.
* [Dataset.rar](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Dataset.rar) - A .zip file which contains the dataset of the images used for train and validation of our model.
* [Tensorboard_logs.zip](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Tensorboard_logs.zip) - A .zip file which contains the [tensorboard](https://www.tensorflow.org/tensorboard) files of the training and validation of both [Custom_CNN.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Custom_CNN.ipynb) and [Transfer_Learning](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Transfer_Learning.ipynb) models.
* [Transfer_Learning.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Transfer_Learning.ipynb) - A Jupyter notebook file where we used [TensorFlow Hub](https://www.tensorflow.org/hub) in order to perform [transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning).
* [requirements.txt](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/requirements.txt) - A .txt file with all the virtual environment requirements to run all the above Jupiter notebook files.

**The first file** "Custom_CNN" is a Convolutional Neural Network developed by me and the main idea is the development of a CNN which will be used in on-line free cameras in Greece, in order to identify forest fires. The train procedure of this CNN was performed with data collected from the internet and the main classes are two: Images with forest fires and images with forests not on fire. Furthemore, the optimization of the time of prediction is important, and the train time is one-time action. Train size: with fire-812 photos, without fire-760 photos. Validation size: with fire-190 photos, without fire-190 photos. Default size of photos:250x250. The depiction of the composition of the specific CNN is the following:

![cnn](https://user-images.githubusercontent.com/75940880/123519158-43814480-d6b2-11eb-8b61-1874fefd2272.png)


![cnn2](https://user-images.githubusercontent.com/75940880/123519380-94456d00-d6b3-11eb-9521-1e4742f9cf91.PNG)


We can observe that the input size of photos is 250x250x3 in order to maintain the maximun size of the image. During experimentation, different stride, batch size, number of neurons and hyperparameters values were used, finalizing the model with the specs provided. 
More detailed information about the CNN is provided through the tensorboard. In this repo you may find the files of training and validation, having as main overview graphs the following:


**The second file** "Transfer_Learning" is a Convolutional Neural Network that uses the tensorflow API to download the mobilenet_v2 version 4 headless and we set the Dense layer the only trainable layer with the classes of the data. In our case the classes are fire and no fire and the CNN will be trained on our data only at the level of the Dense layer. That means that it was trained with a large amoutn of data, "learned" how to extract features of images and classify them and now we change the classification step with our own data, without changing the weights in the other layers. Further documentation regarding mobilenet: [TensorFlow Hub](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4), [Review of mobilenet structure](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c)



