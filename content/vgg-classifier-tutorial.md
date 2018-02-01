Title: Make your own image classifier with Keras
Date: 2016-03-30
Category: Tutorial
Author: Gregory Senay


This tutorial explains how to create an image classifier, using a model pre-trained on ImageNet and fine-tuned over your own dataset.

## Prepare your dataset
-----
Here, I will explain how to prepare your image dataset following an existing simple one: Caltech101.
Caltech101 is a dataset of 101 categories (crocodile, camera, plane, soccer ball...) with 40 to 800 images per category, the image size are different with an average of 200x300 (width=[80;3481] and height=[92;3999]). More information are available on the Caltech website:  [http://www.vision.caltech.edu/Image_Datasets/Caltech101/].
To prepare your dataset, you have to follow this directory architecture:
```
  * Main_Directory/ # is the directory where the directory name is your dataset name
    * Class1Name/ # is a directory with the name of the class 1
      * Image1Class1.jpg # is a picture file of the class 1
      * Image2Class1.jpg
      * Image3Class1.jpg
      * ...
    * Class2Name/
      * Image1Class1.jpg
      * Image2Class1.jpg
      * ...
    * Class3Name/
      * Image1Class1.jpg
      * Image2Class1.jpg
      * ...
    * Class4Name/
    * ...

```

The Caltech101 data is available for download here:  [http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz]. Once you extract the archive with ``tar xvf 101_ObjectCategories.tar.gz)`` you must have this directory architecture:

```
 * 101_ObjectCategories/
 	* Faces/
		 * image_0001.jpg
 		 * image_0002.jpg
		 * image_0003.jpg
		 * ...
 	* Leopards/
		* image_0001.jpg
 		* image_0002.jpg
		* ...
 	* Motorbikes/
		* image_0001.jpg
 		* image_0002.jpg
		* ...
 	* accordion/
	* ...
```

Advice: if you create your own classes, try to have a lot of variabilities for all your classes (i.e. if you have a cat category try to have different cat positions and different cat types, colors...).

Unfortunately, the model you are using below requires a same dimension for all of the images, so we need to load and harmonize the image dataset.

## Browse and load the image directory in python
First, you need to browse the directories with a simple script, and every time you find and an image, you need to normalize it by calling a ``ReadAndNormalizeImage`` function (define below).

```python
def load_data(dim, dirname):
  '''
  - dimension is a tuple like (224,224) corresponding to the height and the width
	- dirname is the main directory like "101_ObjectCategories"
  '''
	X_data = [] # for keeping the image
	y_data = [] # for keeping the image label id
	label_name = [] # for keeping the label dictionary
	label_cpt = 0 # Label id star at index 0
	for class_directory in os.listdir(dirname): # For each class directory
		if os.path.isdir(os.path.join(dirname, class_directory)): # if it's a directory
			for filename in os.listdir(os.path.join(dirname,class_directory)): # for echo file
				img_path = os.path.join(dirname, class_directory, filename) # full path of the file
				if img_path.endswith(".jpg"): # if it's an image -- you can add you own image extension
          normalize_image = ReadAndNormalizeImage(img_path, dim)
					X_data.append(normalize_image)
					y_data.append(label_cpt)
			label_name.append(class_directory)
			label_cpt += 1 # Label id incrementation
			# if label_cpt >= 10: break; # Uncomment to limit to 10 classes
	y_data = np.array(y_data)
	X_data = np.array(X_data, dtype=np.float32)
	label_name = np.array(label_name)
	return X_data, y_data, label_name
```
## Crop, resize and normalize images
``ReadAndNormalizeImage`` function requires ``python-cv2``, if you running a Ubuntu like system, you get it with: `` sudo apt-get install python-cv2``.

The crop, resize and normalize image are split in different functions. The first one crop a square in the center of the image and resize it.
```python
def resize_and_crop_image(img_file, dim):
	'''Takes an image path, crop in the center square if require and resize'''
	img = cv2.imread(img_file) # Read the image with open-cv
	height, width, depth = img.shape # Get the image information
	new_height = height
	new_width = width
	if height > width: # If the image is Landscape
		new_height = width
		height_middle = height / 2
		low_offset = height_middle - int(width/2)
		high_offset = height_middle + int(width/2)
		cropped_img = img[low_offset:high_offset , 0:width] # remove the horizontal sides
		resized_img = cv2.resize(cropped_img,(dim)) # and resize it
		return resized_img
	elif width > height: # If the image is portrait
		new_width = height
		width_middle = width / 2
		low_offset = width_middle - int(height/2)
		high_offset = width_middle + int(height/2)
		cropped_img = img[0:height , low_offset:high_offset] # remove the vertical sides
		resized_img = cv2.resize(cropped_img,(dim)) # resize it
		return resized_img
	else: # It's already a square, no need to crop
		resized_img = cv2.resize(img,(dim)) # just resize it!
		return resized_img
```
There is other ways to do it, like adding some back border on the top and the bottom of the image to keep all the image information (see the opencv copymakeborder function http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=copymakeborder#copymakeborder), but here I simply crop the image in the middle.


 The other function normalize the cropped and resized image, meaning we remove the mean pixel values for each channel (RGB). This normalization was apply on all ImageNet dataset before the initial VGG training and provide a better accuracy.

```python
def ReadAndNormalizeImage(img_file, dim):
	print("Loading", img_file)
	im = resize_and_crop_image(img_file, dim).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	im = im.transpose((2,0,1))
	return im
```

Once you're done you can save the file in the numpy format, to avoid to recreate the dataset every time:

```python
...
np.save("caltech101_X",X_data)
np.save("caltech101_y",y_data)
np.save("caltech101_label",label_name)
return X_data, y_data, label_name
```

So to create your dataset, now you can just call:
```python
X_data, y_data, label_data = load_data((224,224), "101_ObjectCategories")
```
where (224,224) is the final dimension of the image.

Before loading the model, one important operation is <font color='red'>crucial</font>:
<font color='red'>Shuffle you data!!!!!!!!</font> ;).
This is the script to do the shuffle -- but be careful data and label must be shuffle together:
```python
arr = np.arange(len(X_data)) # arr = 0 1 2 3 .... 595
np.random.shuffle(arr) # now arr is shuffled = 45 124 356 18 .... 12
X_data = X_data[arr] # change the order of a shuffled index
y_data = y_data[arr] # same thing of the label to keep the right label!
nb_classes = np.max(y_data)+1 # count how many classes in the dataset
print("Nb classes:", nb_classes)

Y_data = np_utils.to_categorical(y_data, nb_classes) # Last step: convert class vectors to binary class matrices #
# if the class value is 3, then it will become = [ 0 0 0 1 0 0 ....]
```

Now it's time for preparing the model!

Load a pre-trained model
-----

First: you need to download the weights of the model pretrained on ImageNet:
``vgg16_weights.h5`` by following that link: https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view
Here we are using the VGG16 model, if you want more information, please read: http://arxiv.org/abs/1409.1556.

Next, you have to create a ``Keras Sequential`` model (just a stack of layers) with the respect of the original model topology of ``vgg16_weights.h5``.

```python
def VGG16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))
...
```
Then you can load the model:
```python
...
	if weights_path:
		print("Loading weights:",weights_path)
		model.load_weights(weights_path)

	return model

mymodel = VGG16(weights_path='vgg16_weights.h5')

```

Modify VGG for your own dataset
-------
As you can see on the original VGG16 model train on ImageNet, the output number of classes is ``1000``. But you have to change it for your own dataset. To change it, you just need to change the last ``Dense`` (fully connected) layer of the model to fit with our dataset number of classes (``101`` in this tutorial).

## Change the last layer
For changing the number of classes of your model, you need to redefine the layer:

```python
del mymodel.layers[-1]
mymodel.add(Dense(101, activation='softmax'))
```
Nothing more!
Other possibility is to reduce the fully connected layer sizes, by deleting all the last 5 layers and recreate them.
```python
for l in range(0,5):
  del mymodel.layers[-1]
mymodel.add(Dense(256, activation='relu'))
mymodel.add(Dense(0.5))
mymodel.add(Dense(256, activation='relu'))
mymodel.add(Dense(0.5))
mymodel.add(Dense(101, activation='softmax'))
```

## Freeze some layers for training
Because ImageNet is a giant dataset compared to Caltech, the VGG16 pretrained model already knows how to recognize the image details  (http://www.cc.gatech.edu/~hays/compvision/proj6/deepNetVis_small.png), but the model must be tuned to the Caltech dataset. A simple way is to freeze all the convolution layers during the training, meaning the weights of the convolutional layers are never updated. If the convolutional layers are not frozen it would quickly cause an overfitting on the Caltech dataset representations. Only the fully connected layers (``Dense``) are updated. But you can easily make some tests by changing the value of: ``unfreeze_last``. Increasing ``unfreeze_last`` means: more layers with by updated, decreasing ``unfreeze_last`` means: less layers are updated.

In keras, it's easy! You just need to iterate on mymodel.layers to set ``trainable`` attribute of the layer to False or True:

```python
unfreeze_last = 5
for num in range(0,len(mymodel.layers)):
  layer = mymodel.layers[num]
  if num < len(mymodel.layers)-unfreeze_last:
    layer.trainable = False # Freeze for the Firsts
  else:
    layer.trainable = True # Unfreeze for the Lasts
```
## Compile the model
Now, the model can be compile with the categorical cross-entropy loss function  (typically of classification model).
Note that the learning rate need to be low, else the system is not able to adapted the model for your dataset.
If it doesn't work for your own dataset, try to decrease first your the learning rate ``lr`` or next increase it (i.e. 0.05) or change the optimizer by `AdaGrad`, `AdaDelta`, `rmsprop` ...
```python
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
mymodel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
```
To change the optimizer,  you can compile the model with:
```python
mymodel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# or
mymodel.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Train the model
With keras, one function is enough for doing the training, with couple of parameters:
* X_data, Y_data are the dataset and the label
* batch_size=64 is the batch size, if you have a GPU with a small memory size reduce it, with 12Gb you can easily increase to 128
* validation_split=0.1, is the part of the data used for validate your model, this part is very important, you can only say your model is good enough if the accuracy of the validation is good!
* shuffle=True, to shuffle the data at each epoch

And at the end save the new model weights.

```python
mymodel.fit(X_data, Y_data,
      batch_size=64,
      validation_split=0.1,
      nb_epoch=300,
      shuffle=True)

mymodel.save_weights('my_model_weights.h5')
```

If you follow this tutorial, you can quickly achieve an accuracy above 85% on caltech101 in 30 epochs, but don't hesitate to play with the different parameters of the model, like the batch_size, learning_rate or the optimizer.
Or change the model architecture, per example:
- Keep the convolutional layers frozen and increase the number of fully connected layers, but in this case apply a very low learning rate.
- Reduce the size of the fully connected layers (`Dense`), originally this model is design of ImageNet with 1000 classes so the fully connected layers are huge, Caltech101 has 101 so the fully connecter layers can be reduce.
- Unfroze all the layers, and also apply in this case a low learning rate
All of this, to improve the final validation accuracy.

## Test our model
For testing your model, first load and normalize the image (`cat.jpg`) and reshape it in a batch shape.
If you have 4 images, the total shape will be (4,3,224,224).
Then,
```python
my_img = ReadAndNormalizeImage("cat.jpg", (224,224))
my_img = my_img.reshape((1,3,224,224))
mymodel.predict(my_img, batch_size=1, verbose=0)
```
For extraction the class id, use the argmax function
```python
prediction = np.argmax(out)
print(prediction)
```
Or if you want to have the class names, use the `label_data` array:
```python
prediction_class = label_data[np.argmax(out)]
print(prediction_class)
```

-Gregory Senay
