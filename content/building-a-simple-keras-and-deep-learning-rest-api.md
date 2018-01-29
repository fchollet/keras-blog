Title: Building a simple Keras + deep learning REST API
Date: 2018-01-29
Category: Tutorials
Author: Adrian Rosebrock

_This is a guest post by Adrian Rosebrock. Adrian is the author of [PyImageSearch.com](https://www.pyimagesearch.com/),
a blog about computer vision and deep learning. Adrian recently finished authoring_
[Deep Learning for Computer Vision with Python](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)_,
a new book on deep learning for computer vision and image recognition using Keras._

In this tutorial, we will present a simple method to take a Keras model and deploy it as a REST API.

The examples covered in this post will serve as a template/starting point for building your own deep
 learning APIs &mdash; you will be able to extend the code and customize it based on how scalable
 and robust your API endpoint needs to be.

Specifically, we will learn:

- How to (and how not to) load a Keras model into memory so it can be efficiently used for inference
- How to use the Flask web framework to create an endpoint for our API
- How to make predictions using our model, JSON-ify them, and return the results to the client
- How to call our Keras REST API using both cURL and Python

By the end of this tutorial you'll have a good understanding of the components (in their simplest
form) that go into a creating Keras REST API.

Feel free to use the code presented in this guide as a starting point for your own deep learning
REST API.

**Note: The method covered here is intended to be instructional. It is _not_ meant to be
production-level and capable of scaling under heavy load. If you're interested in a more advanced
Keras REST API that leverages message queues and batching, [please refer to this tutorial](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/).**

---

## Configuring your development environment

We'll be making the assumption that Keras is already configured and installed on your machine.
If not, please ensure you install Keras using the [official install instructions](https://keras.io/#installation).

From there, we'll need to install [Flask](http://flask.pocoo.org/) (and its associated
dependencies), a Python web framework, so we can build our API endpoint. We'll also need
[requests](http://docs.python-requests.org/en/master/) so we can consume our API as well.

The relevant `pip` install commands are listed below:

```sh
$ pip install flask gevent requests pillow
```

---

## Building your Keras REST API

Our Keras REST API is self-contained in a single file named `run_keras_server.py`. We kept the
installation in a single file as a manner of simplicity &mdash; the implementation can be easily
modularized as well.

Inside `run_keras_server.py` you'll find three functions, namely:

- `load_model`: Used to load our trained Keras model and prepare it for inference.
- `prepare_image`: This function preprocesses an input image prior to passing it through our
 network for prediction. If you are not working with image data you may want to consider changing
 the name to a more generic `prepare_datapoint` and applying any scaling/normalization you may need.
- `predict`: The actual endpoint of our API that will classify the incoming data from the request
and return the results to the client.

The full code to this tutorial can be found [here](https://github.com/jrosebr1/simple-keras-rest-api).

```python
# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
```

Our first code snippet handles importing our required packages and initializing both the Flask
application and our `model`.

From there we define the `load_model` function:

```python
def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = ResNet50(weights="imagenet")
```

As the name suggests, this method is responsible for instantiating our architecture and loading our
weights from disk.

For the sake of simplicity, we'll be utilizing the ResNet50 architecture which has been pre-trained
on the ImageNet dataset.

If you're using your own custom model you'll want to modify this function to load your
architecture + weights from disk.

Before we can perform prediction on any data coming from our client we first need to prepare and
preprocess the data:

```python
def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image
```

This function:

- Accepts an input image
- Converts the mode to RGB (if necessary)
- Resizes it to 224x224 pixels (the input spatial dimensions for ResNet)
- Preprocesses the array via mean subtraction and scaling

Again, you should modify this function based on any preprocessing, scaling, and/or normalization
you need prior to passing the input data through the model.

We are now ready to define the `predict` function &mdash; this method processes any requests to the
`/predict` endpoint:

```python
@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)
```

The `data` dictionary is used to store any data that we want to return to the client. Right now
this includes a boolean used to indicate if prediction was successful or not &mdash; we'll also use
this dictionary to store the results of any predictions we make on the incoming data.

To accept the incoming data we check if:

- The request method is POST (enabling us to send arbitrary data to the endpoint, including images,
 JSON, encoded-data, etc.)
- An `image` has been passed into the `files` attribute during the POST

We then take the incoming data and:

- Read it in PIL format
- Preprocess it
- Pass it through our network
- Loop over the results and add them individually to the `data["predictions"]` list
- Return the response to the client in JSON format

If you're working with non-image data you should remove the `request.files` code and either parse
the raw input data yourself or utilize `request.get_json()` to automatically parse the input data
to a Python dictionary/object. Additionally, consider giving [following tutorial](https://scotch.io/bar-talk/processing-incoming-request-data-in-flask)
a read which discusses the fundamentals of Flask's `request object`.

All that's left to do now is launch our service:

```python
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()
```

First we call `load_model` which loads our Keras model from disk.

The call to `load_model` is a blocking operation and prevents the web service from starting until
the model is fully loaded. Had we not ensured the model is fully loaded into memory and ready for
inference prior to starting the web service we could run into a situation where:

1. A request is POST'ed to the server.
2. The server accepts the request, preprocesses the data, and then attempts to pass it into the model
3. _...but since the model isn't fully loaded yet, our script will error out!_

When building your own Keras REST APIs, ensure logic is inserted to guarantee your model is loaded
and ready for inference _prior_ to accepting requests.

---

## How to _not_ load a Keras model in a REST API

You may be tempted to load your model _inside_ your `predict` function, like so:

```python
...
	# ensure an image was properly uploaded to our endpoint
	if request.method == "POST":
		if request.files.get("image"):
			# read the image in PIL format
			image = request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# load the model
			model = ResNet50(weights="imagenet")

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []
...
```

This code implies that the `model` will be loaded _each and every time_ a new request comes in.
This is incredibly inefficient and can even cause your system to run out of memory.

If you try to run the code above you'll notice that your API will run considerably slower
(especially if your model is large) &mdash; this is due to the _significant_ overhead in both I/O
and CPU operations used to load your model for _each new request_.

To see how this can easily overwhelm your server's memory, let's suppose we have _N_ incoming
requests to our server at the same time. This implies there will be _N_ models loaded into
memory...again, at the same time. If your model is large, such as ResNet, storing _N_ copies of the
model in RAM could easily exhaust the system memory.

To this end, try to avoid loading a new model instance for every new incoming request unless you
have a very specific, justifiable reason for doing so.

**Caveat:** We are assuming you are using the default Flask server that is single threaded. If you
deploy to a multi-threaded server you could be in a situation where you are *still* loading
multiple models in memory even when using the "more correct" method discussed earlier in this post.
If you intend on using a dedicated server such as Apache or nginx you should consider making
your pipeline more scalable, [as discussed here](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/).

---

## Starting your Keras Rest API

Starting the Keras REST API service is easy.

Open up a terminal and execute:

```sh
$ python run_keras_server.py
Using TensorFlow backend.
 * Loading Keras model and Flask starting server...please wait until server has fully started
...
 * Running on http://127.0.0.1:5000
```

As you can see from the output, our model is loaded _first_ &mdash; after which we can start our
Flask server.

You can now access the server via `http://127.0.0.1:5000`.

However, if you were to copy and paste the IP address + port into your browser you would see the
following image:

![keras api 404](img/simple-keras-rest-api/keras_api_404.png)

The reason for this is because there is no index/homepage set in the Flask URLs routes.

Instead, try to access the `/predict` endpoint via your browser:

![keras api 404](img/simple-keras-rest-api/keras_api_method_not_allowed.png)

And you'll see a "Method Not Allowed" error. This error is due to the fact that your browser is
performing a GET request, but `/predict` only accepts a POST (which we'll demonstrate how to
perform in the next section).

---

## Using cURL to test the Keras REST API

When testing and debugging your Keras REST API, consider using [cURL](https://curl.haxx.se/)
(which is a good tool to learn how to use, regardless).

Below you can see the image we wish to classify, a _dog_, but more specifically a _beagle_:

![dog](img/simple-keras-rest-api/dog.jpg)

We can use `curl` to pass this image to our API and find out what ResNet thinks the image contains:

```sh
$ curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
{
  "predictions": [
    {
      "label": "beagle",
      "probability": 0.9901360869407654
    },
    {
      "label": "Walker_hound",
      "probability": 0.002396771451458335
    },
    {
      "label": "pot",
      "probability": 0.0013951235450804234
    },
    {
      "label": "Brittany_spaniel",
      "probability": 0.001283277408219874
    },
    {
      "label": "bluetick",
      "probability": 0.0010894243605434895
    }
  ],
  "success": true
}
```

The `-X` flag and `POST` value indicates we're performing a POST request.

We supply `-F image=@dog.jpg` to indicate we're submitting form encoded data. The `image` key is
then set to the contents of the `dog.jpg` file. Supplying the `@` prior to `dog.jpg` implies we
would like cURL to load the contents of the image and pass the data to the request.

Finally, we have our endpoint: `http://localhost:5000/predict`

Notice how the input image is correctly classified as _"beagle"_ with 99.01% confidence. The
remaining top-5 predictions and their associated probabilities and included in the response from
our Keras API as well.

---

## Consuming the Keras REST API programmatically

In all likelihood, you will be both _submitting_ data to your Keras REST API and then _consuming_
the returned predictions in some manner &mdash; this requires we programmatically handle the
response from our server.

This is a straightforward process using the [requests](http://docs.python-requests.org/en/master/)
Python package:

```python
# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "dog.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
	# loop over the predictions and display them
	for (i, result) in enumerate(r["predictions"]):
		print("{}. {}: {:.4f}".format(i + 1, result["label"],
			result["probability"]))

# otherwise, the request failed
else:
	print("Request failed")
```

The `KERAS_REST_API_URL` specifies our endpoint while the `IMAGE_PATH` is the path to our input
image residing on disk.

Using the `IMAGE_PATH` we load the image and then construct the `payload` to the request.

Given the `payload` we can POST the data to our endpoint using a call to `requests.post`.
Appending `.json()` to the end of the call instructs `requests` that:

1. The response from the server should be in JSON
2. We would like the JSON object automatically parsed and deserialized for us

Once we have the output of the request, `r`, we can check if the classification is a success
(or not) and then loop over `r["predictions"]`.

To run execute `simple_request.py`, first ensure `run_keras_server.py` (i.e., the Flask web server)
is currently running. From there, execute the following command in a separate shell:

```sh
$ python simple_request.py
1. beagle: 0.9901
2. Walker_hound: 0.0024
3. pot: 0.0014
4. Brittany_spaniel: 0.0013
5. bluetick: 0.0011
```

We have successfully called the Keras REST API and obtained the model's predictions via Python.

---

In this post you learned how to:

- Wrap a Keras model as a REST API using the [Flask web framework](http://docs.python-requests.org/en/master/)
- Utilize cURL to send data to the API
- Use Python and the [requests](http://docs.python-requests.org/en/master/) package to send data
to the endpoint and consume results

The code covered in this tutorial can he found [here](https://github.com/jrosebr1/simple-keras-rest-api)
and is meant to be used as a template for your own Keras REST API &mdash; feel free to modify it as
you see fit.

Please keep in mind that the code in this post is meant to be _instructional_. It is _not_ mean to
be production-level and capable of scaling under heavy load and a large number of incoming requests.

This method is best used when:

1. You need to quickly stand up a REST API for your Keras deep learning model
2. Your endpoint is not going to be hit heavily

If you're interested in a more advanced Keras REST API that leverages message queues and batching,
please refer to [this blog post](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/).

If you have any questions or comments on this post please reach out to [Adrian from PyImageSearch](https://www.pyimagesearch.com/)
(the author of today's post). For suggestions on future topics to cover, please find
[Francois on Twitter](https://twitter.com/fchollet).