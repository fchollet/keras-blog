Title: Set up a Keras Deep Learning environment from scratch
Date: 2016-03-09
Category: Tutorials
Author: Fabien Lavocat

This article will show you a step-by-step setup of a basic Linux environment to develop with Keras. The first step will be to setup an Ubuntu desktop machine, but you can use any Linux distribution you want. Then we will install all the packages needed to run Keras and all its dependencies. We will finish by an optional install of [PyCharm](https://www.jetbrains.com/pycharm/) that you can use as Python developer IDE.

## Install Ubuntu

Ubuntu is one of the most popular Linux distribution, that's why we take this example to setup our environment. Obviously you can use any Linux OS you want. You can even use Keras on your Mac. We choose **Ubuntu version 14.04.4 LTS 64 bits** because most of the libraries have been developed and tested on this version. The 'LTS' meaning Long Term Support, using this version, you have an assurance that it will work for a long time.

First step is for you to download **Ubuntu 14.04.4 LTS 64 bits** from the official website: http://www.ubuntu.com/download/desktop. Follow their instructions to run the installer on your computer (from a CD or a USB key).

![Ubuntu Logo](/img/setup-keras/install/01-ubuntu-logo.png)

The installer asks you to choose the language of the OS. Then press the **Continue** button.

![Ubuntu - Welcome](/img/setup-keras/install/02-ubuntu-welcome.png)

Make sure you have enough space on your disk and you are connected to the Internet. Check the checkbox **Download updates while installing**, this will save you some time later. We do not need to install third-party software on our machine. We want to keep it as minimal as possible. Then press the **Continue** button.

![Ubuntu - Preparing to install Ubuntu](/img/setup-keras/install/03-ubuntu-preparing.png)

This part is very specific to your own computer. Make sure to not erase sensitive data. Or if you have a dual boot with another OS, try not to erase anything from the other partitions. Then press the **Install Now** button.

![Ubuntu - Installation type](/img/setup-keras/install/04-ubuntu-install-type.png)

Quick confirmation, you still have time to go back and make changes to your partitioning. Then press the **Continue** button.

![Ubuntu - Write the changes to disks](/img/setup-keras/install/05-ubuntu-erase-disk.png)

Select the city you are located to have the correct timezone. Then press the **Continue** button.

![Ubuntu - Where are you?](/img/setup-keras/install/06-ubuntu-location.png)

Select your keyboard layout. Then press the **Continue** button.

![Ubuntu - Keyboard](/img/setup-keras/install/07-ubuntu-keyboard.png)

Enter your name, the name of your computer, a username and a password. Then press the **Continue** button.

![Ubuntu - User Account](/img/setup-keras/install/08-ubuntu-user-account.png)

Install in progress...

![Ubuntu - Install in progress](/img/setup-keras/install/09-ubuntu-install.png)

You are one reboot away from the Ubuntu desktop. Then press the **Restart Now** button.

![Ubuntu - Restart](/img/setup-keras/install/10-ubuntu-restart.png)

Login using the credentials you've entered in the installation process.

![Ubuntu - Login](/img/setup-keras/install/11-ubuntu-login.png)

Congratulation, you're on the Ubuntu desktop ready to install Keras. You might have some updates to install. Then press the **Install Now** button.

![Ubuntu - Updates](/img/setup-keras/install/12-ubuntu-updates.png)

## Install Keras and its dependencies

First, you must open a **Terminal** and run the script bellow, line by line to install all the requirements of Keras and then Keras itself.

```python
# Python is installed by default with Ubuntu. 
# Let's check the current version of Python
python --version
```

On my computer I have the version 2.7.6 of python. This will be enough for our dev environment.

![Ubuntu - Python version](/img/setup-keras/terminal/01-python-version.png)

```python
# Update all the current packages to their latest version
sudo apt-get update

# A lot of Python packages are available using PIP (Python Package Index)
# Install the package python-pip
sudo apt-get -y install python-pip

# Packages needed to build Scipy and Numpy
sudo apt-get install python2.7-dev build-essential python-dev gfortran

# Other required packages for Scipy and Numpy
sudo apt-get install libatlas-dev liblapack-dev

# Install numpy and scipy. Those two packages might take a while to install
sudo pip install numpy
sudo pip install scipy

# Install git to get the latest version of the next package
sudo apt-get install git

# Install Theano (backend for Keras)
# The other option is to TensorFlow (from Google) but we will see that option in a future post
sudo pip install git+git://github.com/Theano/Theano.git

# Install Keras
sudo pip install keras
```

Here you are! You've installed the minimal packages to run your first script using Keras. To make sure everything works, keep your terminal open and enter the following commands:

```python
# Run the python shell
python

# This line will import the keras library
import keras
```

If you don't see any error, it means you have susccess fully install Keras and all its dependencies.

![Ubuntu - Import Keras](/img/setup-keras/terminal/02-import-keras.png)

Now, let's run your first Neural Network with Keras. The following line will download an example script from the official Keras GitHub and run it in Python. This script is a Neural Network that learns how to perform additions. 

```
curl -ssl https://github.com/fchollet/keras/raw/master/examples/addition_rnn.py | python
```

![Run example](/img/setup-keras/terminal/03-run-example.png)

After 4 training iterations...

![Iteration 4](/img/setup-keras/terminal/04-iteration-4.png)

After 20 training iterations...

![Iteration 4](/img/setup-keras/terminal/05-iteration-20.png)

## Install PyCharm

PyCharm Community is a free and lightweight IDE for Python that will allow you writing, running and debugging your Python project very easily. This step is optional and if you have another IDE that you like to use, you can skip this step.

![PyCharm](/img/setup-keras/pycharm/01-ide.jpg)

First, we must install the dependency required to install PyCharm, the Java JRE. Open your terminal and run the following command to install the last version of the Oracle JRE.

```
sudo apt-get install default-jre
```

To make sure the Java runtime was properly installed, run the following command to get the current version of Java installed on the machine.

```
java -version
```

![Java version](/img/setup-keras/pycharm/02-java-version.png)

Download **PyCharm Community** on your computer from https://www.jetbrains.com/pycharm/
Then, open your terminal and run the following command to create the folder /opt/PyCharm

```
sudo mkdir -p /opt/PyCharm
```

Then, extract the file you've downloaded to this folder using the following command.

```
sudo tar -zxvf pycharm-community-5.0.4.tar.gz --strip-components 1 -C /opt/PyCharm
```

Now, you can open PyCharm using the following command.

```
sudo /opt/PyCharm/bin/pycharm.sh
```

![Run PyCharm](/img/setup-keras/pycharm/03-run-pycharm.png)

Congratulation on following this guide, you are now ready to start your first Deep Learning project using Keras on top of Theano.