# Flappy Bot

A simple Flappy Bird AI, which uses a neural network to classify ingame screenshots and decide on key presses

> Note: this repository uses the flappy bird game of https://github.com/LeonMarqs/Flappy-bird-python as a submodule.

## Requirements

**Python modules:**
- numpy
- Image
- ImageFilter
- os
- uuid

**OS specific packages:**
- For MacOS:
  - tensorflow-macos
  - tensorflow-metal
- For Linux/Windows:
  - tensorflow

## Quick Setup

### **Tensorflow Installation Guide on Windows:**

If you want to use Tensorflow with your GPU follow this installation guide:

[Installation Guide with GPU](TF_GPU_WIN.md)

Otherwise, if you don't care about your computer burning and your fans blowing like a hurricane (if you have one) by using your CPU, you just need to install tensorflow with the following command:

        $ pip install tensorflow



## The game

<img src="img/game_exemple.gif" width="300px" height="400px">

In this game, the goal is to control a bird that has to go throught pipes. If you touch the ground or a pipe, you lose the game. In this game version, to jump you should press space bar. 

<img src="img/set_ia-mode_false.png" width="425px" height="75px">

If you want to play to the game, you should go in the Flappy-bird-python-dataset folder and set **ia_mode** value to false. Then you should launch the file.



## Training Your Own

TODO
