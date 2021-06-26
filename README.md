# Deep-Learning
This repository consists of two .ipynb files, in the scope of the second semester "Deep Learning" exersice in Master in AI in NSCR Demokritos.The first file "Custom_CNN" is a Convolutional Neural Network developed by me and the main idea is the development of a CNN which will be used in on-line free cameras in Greece, in order to identify forest fires. The train procedure of this CNN was performed with data collected from the internet and the main classes are two: Images with forest fires and images with forests not on fire. Furthemore, the optimization of the time of prediction is important, and the train time is one-time action. Train size: with fire-812 photos, without fire-760 photos. Validation size: with fire-190 photos, without fire-190 photos. Default size of photos:250x250. The depiction of the composition of the specific CNN is the following:

![cnn](https://user-images.githubusercontent.com/75940880/123519158-43814480-d6b2-11eb-8b61-1874fefd2272.png)


Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 248, 248, 64)      1792      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 124, 124, 64)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 122, 122, 64)      36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 61, 61, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 59, 59, 64)        36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 29, 29, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 27, 27, 64)        36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 13, 13, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 10816)             0         
_________________________________________________________________
dropout (Dropout)            (None, 10816)             0         
_________________________________________________________________
dense (Dense)                (None, 1)                 10817     
=================================================================
Total params: 123,393
Trainable params: 123,393
Non-trainable params: 0


We can observe that the input size of photos is 250x250x3 in order to maintain the maximun size of the image. During experimentation, different stride, batch size, number of neurons and hyperparameters values were used, finalizing the model with the specs provided. 
