Title: Introducing Keras 1.0
Date: 2016-04-11
Category: News
Author: Francois Chollet

Keras was initially released a year ago, late March 2015. It has made tremendous progress since, both on the development front, and as a community.

But continuous improvement isn't enough. A year of developing Keras, using Keras, and getting feedback from thousands of users has taught us a lot. To the point that we are now able to redesign it better than we could have the first time around.

And so today we are releasing Keras 1.0. It isn't a patch on top of the previous version, it is actually a re-writing of Keras nearly from scratch. It maintains backwards compatibility while introducing a series of major features, made possible by a better design under the hood.

Simplicity and accessibility have always been the targets guiding the Keras development efforts. The purpose of Keras is to make deep learning accessible to as many people as possible, by providing a set of "Lego blocks" for building Deep Learning models in a fast and simple way. Keras 1.0 pushes even further in that same direction.

The most significant feature introduced today is the functional API, a new way to define your Keras models. Get started with the functional API with [this short guide](http://keras.io/getting-started/functional-api-guide/). If you are new to Keras, first read the ["30 seconds to Keras" introduction](http://keras.io/#getting-started-30-seconds-to-keras), then read [this overview of the Sequential model](http://keras.io/getting-started/sequential-model-guide/).


## New features

- The functional API: a simpler and more powerful way to define complex deep learning models.

- Better performance. Compilation times are lower. All RNNs now come in 2 different implementations to choose from, allowing you to get maximum performance across widely different tasks and setups. And Theano RNNs can now be unrolled, yielding up to a 25% speed-up.

- Modular metrics. You can know monitor arbitrary lists of metrics on arbitrary endpoints of your Keras models.

- An even better user experience. The code has been rewritten from scratch with the end user in mind at all stages. A great library UX has two components: simple, intuitive APIs (the kind that are easy to memorize), and the ability to return sensible, easy to grok error messages whenever faced with a user error.

- Improved Lambda layers.


...and much more.

## Installation

You can install the new version from PyPI:

```sh
pip install keras --upgrade
```

Or from the master branch on Gihub:

```sh
git clone https://github.com/fchollet/keras.git
cd keras
python setup.py install
```

## Porting custom Keras layers

Because the Keras internals have changed, custom Keras layers will not work with the new version. However you can port them to the new version in just a minute. Simply follow the instructions in [this guide](https://github.com/fchollet/keras/wiki/Porting-your-custom-layers-from-Keras-0.3-to-Keras-1.0).