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

**The first file** [Cameras.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Cameras.ipynb), is a list of cameras in greece, which visualize mountains and vilages with green and forests in Greece and is used to test our code but also show an aspect of usage as production code. Inside this file there is a visualization of the cameras, where we take a real-time snapshot in order for us to use it for prediction with our trained models, in an effort to use it to solve a real world problem. The code we used is the library [requests](https://realpython.com/python-requests/), in order to obtain a snapshot of the url and plot the photos in the code with [matplotlib](https://matplotlib.org/).

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
* The MaxPooling hyper parameters are the following: pool_size=(2, 2), strides=2, [padding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)='valid'. We chose stride of 2 in order to avoid overfitting and cocluded to this value through our experiments. We also set the padding = "valid" because is a dynamic solution and according to documentation **"The resulting output, when using the "valid" padding option, has a spatial shape (number of rows or columns) of: output_shape = math.floor((input_shape - pool_size) / strides) + 1 (when input_shape >= pool_size)"**.
* We chose to add a dropout layer with value 20% due to overfitting in our experiments.
* The last layers are the Flatten (fully connected) layer and the Dense of 1 neuron where we classify our output to "fire" or "no fire". 
* Our compiling hyperparameters are loss='binary_crossentropy',optimizer = '[adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)',metrics=['accuracy',F1_Score()]
* We set the fit (train) of the model and added to our callbacks the tensorboard, the time monitor and the auto-save of the best weights. The train procedure of this CNN was performed with [data](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Dataset.rar) collected from the internet and the main classes are two: Images with forest fires and images with forests not on fire. Furthemore, the optimization of the time of prediction is important, and the train time is one-time action. Train size: with fire-760 photos, without fire-760 photos. Validation size: with fire-190 photos, without fire-190 photos. Default size of photos:250x250.
* After the training, which lasted 8000 seconds (aprox 2,22 hours) we began the preprocess of the [Cameras.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Cameras.ipynb) in order to be eligible for our network. **!Note: we could stop at epoch 13 which is the model we selected in the end an the training time was 2200 seconds (aprox 36.6 minutes), but for experimental reasons we set 50 epochs!** For every url we provided, we predict and print real-time if there was detected a fire or not. Also due to possible issues of connection to some cameras, we also print the number of cameras we could not connect to. If there is a fire detected, the url is also printed, in order for the user to examine further.
* In an attempt to provide an explainability of our network, we chose a no-fire photo and the neurons activated in every layer. We can observe this "black box" and why it chose to predict "no fire".
* Continuing our testing, we tried to "photoshop" a fire inside a "no-fire" url. Thankfully, we had no urls with fire, so we chose the last of the list. With python and the library [Pillow](https://pillow.readthedocs.io/en/stable/), we added the photo of the [url](https://wallpaperaccess.com/full/1817829.jpg) (after preprocessing it) and used it to predict what would the output be in a possible fire situation. We printed the output and the photo used for predicting (250x250).
* The output was "fire" but we also wanted to visualize the "why fire?". As before we visualized the activation of the neurons in each layer and we can observe that our model has distincted the fire and not any other pattern in the photo provided!
* Lastly we used a technique to save and load the model. We may also save only the weights of the model.
* A last note is that we also used BatchNormalization after the activation function as stated in the [Deep Learning book by Aaron Courville, Ian Goodfellow, and Yoshua Bengio, Chapter 8](https://www.deeplearningbook.org/contents/optimization.html) in chapter 8.7.1 in page 313, but our scores were worse than before, so we procceded without this technique. Possibly with further analysis, we could use this technique correctly and achieve better results.
* ****Detailed performance metrics will be provided when explaining the file [Tensorboard_logs.zip](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Tensorboard_logs.zip)****

**The third file** [Dataset.rar](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Dataset.rar), is the datase we created with photos from the internet and size of (250x250). We tried to keep a pattern and only select photos with forests in order to have better results.

_We will explain [Tensorboard_logs.zip](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Tensorboard_logs.zip) file last, in order to analyze the performance of both of the models, compare them and analyse the results after having presented both of our models._

**The fifth file** [Transfer_Learning.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Transfer_Learning.ipynb), is a Convolutional Neural Network that uses the tensorflow API to download the mobilenet_v2 version 4. Further documentation regarding mobilenet: [TensorFlow Hub](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4), [Review of mobilenet structure](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c). The main idea to this file are same as the [Custom_CNN.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Custom_CNN.ipynb), but with a few differences in the approach in the code. The most important milestones during the creation of this code, where the following:

* We imported the necessary libraries which we will use. See [requirements.txt](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/requirements.txt) for more info.
* We constructed an object for the f1 metric with the usage of the library [Tensorflow Addons](https://medium.com/tensorflow/introducing-tensorflow-addons-6131a50a3dcf)
* We constructed a Class in order to auto-save the best model during our training and then use it but loading the saved weights to the model. Only the weights are saved and the metric we use for the "best model" decision is the "val_loss". Reference for this technique in this [article](https://stackoverflow.com/questions/61630990/tensorflow-callback-how-to-save-best-model-on-the-memory-not-on-the-disk).
* We set the path and arguments in order to monitor our train and evaluate our model through [tensorboard](https://www.tensorflow.org/tensorboard) and a Class to monitor the training time. We also created an [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) object, but we did not use it due to no hardware or resource limitations.
* We did the preprocess of our data and used [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator). We changed the size of the images to 224x224 as stated in the documentation of mobilenet.
* The main difference between Custom_CNN and mobilnet is that now, we do not need to train the whole model, because it is already trained with the dataset Imagenet, so it is a very good feature extractor! We have to only train the last (Dense layer) with our dataset and set the rest network to trainable=False. That means that it was trained with a large amoutn of data, "learned" how to extract features of images and classify them and now we change the classification step with our own data, without changing the weights in the other layers. This can be observed clearly in the step 35 of our code.
* Architecture of the model:

![image](https://user-images.githubusercontent.com/75940880/124360776-4b059800-dc34-11eb-85ed-a45d670868d5.png)

![image](https://user-images.githubusercontent.com/75940880/124360768-3b864f00-dc34-11eb-8066-6ad5f3119749.png)


* After training the last layer to our dataset, which lasted 600 seconds (aprox 10 minutes for 10 epochs), we visualized a sample of the train images and their labels as predicted from our network (green color if correct - red if not correct).
* We then use our network to predict if there is a fire in the [Cameras.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Cameras.ipynb) and also used the same technique of "photoshop" as in [Custom_CNN.ipynb](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Custom_CNN.ipynb).
* Lastly, we used a save and load technique of our model, and as before we could also save only the weights but we saved the whole model.

**The fourth file** [Tensorboard_logs.zip](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Tensorboard_logs.zip) contains all the training and validation information, but we will use the main graphs to visualize, analyse, explain, evaluate and compare our 2 models.

* Firstly, let's see the train and validation of [Custom_CNN model](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Custom_CNN.ipynb). The metrics used are accuracy, f1 score and loss.

### ****Green is the train and gray is validation****

## **Accuracy**:

![image](https://user-images.githubusercontent.com/75940880/124359635-7dac9200-dc2e-11eb-9b7a-38cd55a77693.png)

## **F1 Score**:

![image](https://user-images.githubusercontent.com/75940880/124359680-b187b780-dc2e-11eb-99a8-2f9159cf7283.png)

## **Loss**:

![image](https://user-images.githubusercontent.com/75940880/124359724-df6cfc00-dc2e-11eb-900f-ab1084145ddc.png)

**Comments regarding [Custom_CNN model](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Custom_CNN.ipynb) performance:**
* **Accuracy** is a metric to measure the performance of our model, but not a reliable one. Without analysing in detail the accuracy, we simply state that the model we selected and use, the one of the epoch 13, has train accuracy 98.75% and validation accuracy of 92.89%.
* **F1 Score** is a reliable metric and as we can see in our model, the one of the epoch 13, has train f1 score 98.75% and validation of 93.09%.
* **Loss** is the metric which we chose to select our "best model" (validation loss) and is the minimun on epoch 13 with values on train 0.044 and validation 0.1819.

**A result during inspecting the data of train and validation is that after the epoch 13, the model began to overfit to the train data. When we used BatchNormalization, the model did not overfit but in every epoch it remain in the final values in all metrics. The loss of this state while using BatchNormalization was 0.3230 and is the reason why we removed it from our code after experimenting in different positions between the layers.**

* Secondly, let's see the train and validation of [Transfer_Learning model](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Transfer_Learning.ipynb). The metrics used are accuracy, f1 score and loss.

### ****Orange is the train and blue is validation****

## **Accuracy**:

![image](https://user-images.githubusercontent.com/75940880/124360208-396ec100-dc31-11eb-85e7-e73ced694a7a.png)

## **F1 Score**:

![image](https://user-images.githubusercontent.com/75940880/124360245-61f6bb00-dc31-11eb-97c1-42d6e0b70701.png)

## **Loss**:

![image](https://user-images.githubusercontent.com/75940880/124360268-79ce3f00-dc31-11eb-80e8-632a535ca406.png)

**Comments regarding [Transfer_Learning model](https://github.com/Andreas-Stefopoulos/Deep-Learning/blob/main/Transfer_Learning.ipynb) performance:**
* **Accuracy** is a metric to measure the performance of our model, but not a reliable one. Without analysing in detail the accuracy, we simply state that the model we selected and use, the one of the epoch 10, has train accuracy 98.87% and validation accuracy of 98.68%.
* **F1 Score** is a reliable metric and as we can see in our model, the one of the epoch 10, has train f1 score 98.87% and validation of 98.68%.
* **Loss** is the metric which we chose to select our "best model" (validation loss) and is the minimun on epoch 10 with values on train 0.01084 and validation 0.05148.

**A result during inspecting both of the train-val data, we may clearly see that the mobilenet is considerably superior in comparison with the CNN created by us. This can be seen while plotting both train-validation data clearly**

### ****Green is the train and gray is validation Custom CNN****
### ****Orange is the train and blue is validation for mobilenet****

![image](https://user-images.githubusercontent.com/75940880/124360566-3a085700-dc33-11eb-9230-e5ec58dab5c2.png)

The result of all the above is that even though our custom network is not as good as the mobilenet, it is reliable enough to recognize a fire in a forest. The main difference is that our custom network might not classify a small fire in an image as "fire", in contrast with mobilenet which can classify smaller fires (we came to this result during experimentation).
