# Deep-Learning
This repository consists of .ipynb files, in the scope of the second semester "Deep Learning" exersice in Master in AI in NSCR Demokritos.

Overview of the files:
* [Cameras.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Cameras.ipynb) - A Jupyter notebook file where we visualize cameras in Greece, which we use to predict if there is a fire or not.
* [Custom_CNN.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Custom_CNN.ipynb) - A Jupyter notebook file where we created a Convolutional Neural Network from scratch, which is used to predict in images if there is a fire or not.
* [Dataset.rar](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Dataset.rar) - A .zip file which contains the dataset of the images used for train and validation of our model.
* [Tensorboard_logs.zip](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Tensorboard_logs.zip) - A .zip file which contains the [tensorboard](https://www.tensorflow.org/tensorboard) files of the training and validation of both [Custom_CNN.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Custom_CNN.ipynb) and [Transfer_Learning](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Transfer_Learning.ipynb) models.
* [Transfer_Learning.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Transfer_Learning.ipynb) - A Jupyter notebook file where we used [TensorFlow Hub](https://www.tensorflow.org/hub) in order to perform [transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning).
* [requirements.txt](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/requirements.txt) - A .txt file with all the virtual environment requirements to run all the above Jupiter notebook files.

Detailed steps followed and references used for development of the files:

**The first file** [Cameras.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Cameras.ipynb), is a list of cameras in greece, which visualize mountains and vilages with green and forests in Greece and is used to test our code but also show an aspect of usage as production code. Inside this file there is a visualization of the cameras, where we take a real-time snapshot in order for us to use it for prediction with our trained models, in an effort to use it to solve a real world problem. The code we used is the library [requests](https://realpython.com/python-requests/), in order to obtain a snapsot of the url and plot the photos in the code with [matplotlib](https://matplotlib.org/).

**The second file** [Custom_CNN.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Custom_CNN.ipynb), is a Convolutional Neural Network developed by me and my team in the master and the main idea is the development of a CNN which will be used with [Cameras.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Cameras.ipynb), in order to identify forest fires. The most important milestones during the creation of this code, where the following:

* We imported the necessary libraries which we will use. See [requirements.txt](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/requirements.txt) for more info.
* We constructed a Class for the f1 score metric, with the help of this [article](https://stackoverflow.com/questions/64474463/custom-f1-score-metric-in-tensorflow)
* We constructed a Class in order to auto-save the best model during our training and then use it but loading the saved weights to the model. Only the weights are saved and the metric we use for the "best model" decision is the "val_loss". Reference for this technique in this [article](https://stackoverflow.com/questions/61630990/tensorflow-callback-how-to-save-best-model-on-the-memory-not-on-the-disk).
* We set the path and arguments in order to monitor our train and evaluate our model through [tensorboard](https://www.tensorflow.org/tensorboard) and a Class to monitor the training time. We also created an [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) object, but we did not use it due to no hardware or resource limitations.
* We did the preprocess of our data and used [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator). We kept the original size of our photos in the dataset (250x250) in order to have more accurate training.
* We began constructing our Custom CNN by firstly setting the input shape to (250,250,3). Then we added the layers, the architecture of which can be seen in the following images:  

![image](https://user-images.githubusercontent.com/75940880/124357191-eaba2a80-dc22-11eb-9e39-fcecb290eccd.png) 

![image](https://user-images.githubusercontent.com/75940880/124357249-366cd400-dc23-11eb-874a-e7f768926bce.png)

* The hyperparameters for our 4 Convolution layers are the same and the following:
       filters = 64,
       kernel_size = 3, 
       activation='relu',
       input_shape=input_shape (250,250,3)
* The MaxPooling hyper parameters are the following: pool_size=(2, 2), strides=2, padding='valid'. We chose stride of 2 in order to avoid overfitting and cocluded to this value through our experiments.
* We chose to add a dropout layer with value 20% due to overfitting in our experiments.
* The last layers are the Flatten (fully connected) layer and the Dense of 1 neuron where we classify our output to "fire" or "no fire". 
* Our compiling hyperparameters are loss='binary_crossentropy',optimizer = '[adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)',metrics=['accuracy',F1_Score()]
* We set the fit (train) of the model and added to our callbacks the tensorboard, the time monitor and the auto-save of the best weights.
* After training



The train procedure of this CNN was performed with data collected from the internet and the main classes are two: Images with forest fires and images with forests not on fire. Furthemore, the optimization of the time of prediction is important, and the train time is one-time action. Train size: with fire-812 photos, without fire-760 photos. Validation size: with fire-190 photos, without fire-190 photos. Default size of photos:250x250. The depiction of the composition of the specific CNN is the following:



![cnn](https://user-images.githubusercontent.com/75940880/123519158-43814480-d6b2-11eb-8b61-1874fefd2272.png)


![cnn2](https://user-images.githubusercontent.com/75940880/123519380-94456d00-d6b3-11eb-9521-1e4742f9cf91.PNG)


We can observe that the input size of photos is 250x250x3 in order to maintain the maximun size of the image. During experimentation, different stride, batch size, number of neurons and hyperparameters values were used, finalizing the model with the specs provided. 
More detailed information about the CNN is provided through the tensorboard. In this repo you may find the files of training and validation, having as main overview graphs the following:


**The second file** "Transfer_Learning" is a Convolutional Neural Network that uses the tensorflow API to download the mobilenet_v2 version 4 headless and we set the Dense layer the only trainable layer with the classes of the data. In our case the classes are fire and no fire and the CNN will be trained on our data only at the level of the Dense layer. That means that it was trained with a large amoutn of data, "learned" how to extract features of images and classify them and now we change the classification step with our own data, without changing the weights in the other layers. Further documentation regarding mobilenet: [TensorFlow Hub](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4), [Review of mobilenet structure](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c)



