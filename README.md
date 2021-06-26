# Deep-Learning
This repository consists of two .ipynb files, in the scope of the second semester "Deep Learning" exersice in Master in AI in NSCR Demokritos.The first file "Custom_CNN" is a Convolutional Neural Network developed by me and the main idea is the development of a CNN which will be used in on-line free cameras in Greece, in order to identify forest fires. The train procedure of this CNN was performed with data collected from the internet and the main classes are two: Images with forest fires and images with forests not on fire. Furthemore, the optimization of the time of prediction is important, and the train time is one-time action. Train size: with fire-812 photos, without fire-760 photos. Validation size: with fire-190 photos, without fire-190 photos. Default size of photos:250x250. The depiction of the composition of the specific CNN is the following:

![cnn](https://user-images.githubusercontent.com/75940880/123519158-43814480-d6b2-11eb-8b61-1874fefd2272.png)


![cnn2](https://user-images.githubusercontent.com/75940880/123519380-94456d00-d6b3-11eb-9521-1e4742f9cf91.PNG)


We can observe that the input size of photos is 250x250x3 in order to maintain the maximun size of the image. During experimentation, different stride, batch size, number of neurons and hyperparameters values were used, finalizing the model with the specs provided. 
More detailed information about the CNN is provided through the tensorboard. In this repo you may find the files of training and validation, having as main overview graphs the following:
