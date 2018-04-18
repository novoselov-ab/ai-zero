# AI-Zero

Implementation of an AlphaGo Zero paper in one C++ header file without any dependencies.

To achieve that [include/ai.h](include/ai.h) implements:
* Basic multilayer neural networks
* Multiple input and multiple output layers
* Convolution layers
* MSE and cross entropy loss
* Model serialization
* Extendable optimizer, currently: Adagrad, Adam, SGD
* Basic RL environment: game, player, replay buffer
* Monte Carlo Tree Tearch with policy NN
* Self-play, optimization, evaluation and validation workers

The idea is to have simple, minimal and easy to understand implementation. That means that performance was not a priority. Howeever the whole training process runs in parallel on 4 threads (4 workers) and optimizer also spawns some threads, it was easy to do that without sacrificing simplicity of the project.

Examples include (all single main.cpp files):
* Mnist CNN training
* Gradient checking
* Connect4 game RL training

## Building

Run `premake.bat` to generate solutions. 

It was tested only on windows. But since it's just 2 files of code for every example, it should work out of the box. There is no platform specific code.

