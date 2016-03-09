Title: Setup a new Keras environment
Date: 2016-03-09
Category: News
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

Select the city you are located to have the timezone correct. Then press the **Continue** button.

![Ubuntu - Where are you?](/img/setup-keras/install/06-ubuntu-location.png)

Select your keyboard layout. Then press the **Continue** button.

![Ubuntu - ](/img/setup-keras/install/07-ubuntu-keyboard.png)

Enter your name, the name of your computer, a username and a password. Then press the **Continue** button.

![Ubuntu - ](/img/setup-keras/install/08-ubuntu-user-account.png)

Install in progress...

![Ubuntu - ](/img/setup-keras/install/09-ubuntu-install.png)

You are one reboot away from the Ubuntu desktop. Then press the **Restart Now** button.

![Ubuntu - ](/img/setup-keras/install/10-ubuntu-restart.png)

Login using the credentials you've entered in the installation process.

![Ubuntu - ](/img/setup-keras/install/11-ubuntu-login.png)

Congratulation, you're on the Ubuntu desktop ready to install Keras. You might have some updates to install. Then press the **Install Now** button.

![Ubuntu - ](/img/setup-keras/install/12-ubuntu-updates.png)
