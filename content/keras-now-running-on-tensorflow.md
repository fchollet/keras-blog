Title: Keras, now running on TensorFlow
Date: 2015-12-01
Category: News
Author: Francois Chollet

The purpose of Keras is to be a model-level framework, providing a set of "Lego blocks" for building Deep Learning models in a fast and straightforward way.
Among Deep Learning frameworks, Keras is resolutely high up on the ladder of abstraction.

As such, Keras does not handle itself low-level tensor operations, such as tensor products and convolutions.
Other teams have developed excellent solutions to the tensor manipulation problem, such as [Theano](http://deeplearning.net/software/theano/) (from the LISA/MILA lab of Université de Montréal) and recently [TensorFlow](http://www.tensorflow.org/) (from Google). 

When we started Keras in March 2015, Theano was the natural choice. At the time, it was the only framework with everything we sought: a Python-based interface, autodifferentiation, and the ability to run the same code on CPU and GPU seamlessly. It was also well-optimized and competitively fast.

Since then, there has been a lot of innovation in the symbolic tensor computation space --a lot of it in the footsteps of Theano. Most notably, we've seen two new frameworks appear, Neon from Nervana Systems and TensorFlow from Google. While Neon is the faster framework right now, TensorFlow has the engineering weight of Google behind it and there is no doubt that it will improve considerably over the next few months.

It has become time for Keras to take advantage of these advances. Over the past two weeks, we've abstracted the tensor-manipulation backend of Keras, and we've written two implementations of this backend, one in Theano and the other in TensorFlow. A Neon one might be coming soon as well.

## What this means for you

If you had any models written in vanilla Keras, you can now run them in TensorFlow with no changes on your part. Yes, really. There are a couple of temporary caveats (see "known issues"), but only a small minority of models will be concerned. And of course, you can keep running your models on Theano as you did before.

It also means that any performance improvement on the TensorFlow side henceforth will benefit you and your research. You too, now, as a user of Keras, are riding the Google rocketship.

And at last, it means that you will soon be able to easily export your Keras models to mobile devices, or even to a tractor ([deep learners from Tennessee](https://twitter.com/iamtrask/status/669984166633734144), rejoice), as soon as this support is enabled in open-source TensorFlow. Robotics is likely to be one major field of application of Deep Learning in the coming years.

## Performance

On CPU, here's what performance looks like right now on [some basic examples scripts](https://github.com/fchollet/keras/tree/master/examples). This on a 2.2 GHz Intel Core i7.

In summary: Theano has well-optimized tensor loops compared to TensorFlow, but relies on a poorly-performing CPU convolution operation (of course, few people would actually attempt to train convnets on CPU, although with TensorFlow it wouldn't be too unrealistic).

|       Task                       |     TensorFlow        |      Theano  |
|----------------------------------|-----------------------|--------------|        
| mnist_mlp.py: compilation (s)    |  **0.6**              |      5.9     | 
| mnist_mlp.py: runtime/epoch (s)  |    7.5                |     **6.3**  |
| imdb_lstm.py: compilation (s)    |  39.3                 |   **38.3**   |
| imdb_lstm.py: runtime/epoch (s)  |  283                  |    **123**   | 
| mnist_cnn.py: compilation (s)    |  **0.8**              |       11.4   |
| mnist_cnn.py: runtime/epoch (s)  | **190**               |       3230   |

A similar benchmark on GPU will be added soon.

## Known issues

Due to current limitations of TensorFlow, not all Keras features will work in TensorFlow right now. **However, these limitations are being fixed as we speak, and will be lifted in upcoming TensorFlow releases**. If you need any of the features below, you'll have to wait a little bit before switching to TensorFlow.

- the `dot` mode in `Merge` won't work in TensorFlow.
- Masking in RNNs won't work in TensorFlow **[January 2016 update: it does work now.]**
- When using RNNs in TensorFlow, you will need to explicitely define the number of timesteps per sequence.

There is also one issue that might take a bit more time to understand and fix: the weights of convolutional models saved with Theano can't be successfully loaded in TensorFlow, and reciprocally. We're investigating it right now.


## Try it now

To use the TensorFlow backend, just update Keras, then see [these instructions](http://keras.io/backend/#switching-from-one-backend-to-another).