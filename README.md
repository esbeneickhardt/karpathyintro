# Introduction
In this repository I follow along with Karpathy's deep learning series ["Zero to Hero"](https://karpathy.ai/zero-to-hero.html), and write notes and code.

# Overview
So far I have completed the following parts of his lessons:

1. Micrograd  
  * Description:
    * We build a module from scratch which can do backpropagation and train neural networks
  * Links:
    * [Youtube-Lecture](https://youtu.be/VMj-3S1tku0)
	* [Karpathys-Github](https://github.com/karpathy/micrograd)
  * Files:
    * The outcome from the lecture has been written into the notebooks in the folder /notebooks/micrograd
	* The minimal code to reproduce the micrograd library has been exported from the notebooks to /modules/micrograd

2. Language Modelling: Part 1
  * Description:  
  	* We build a simple bigram character-level language model in pytorch
  * Links:
  	* [Youtube-Lecture](https://www.youtube.com/watch?v=PaCmpygFfXo)
	* [Karpathys-Github](https://github.com/karpathy/makemore)
  * Files:
  	* A walkthrough of all the information given in the lecture has been written to notebook one in the folder /notebooks/bigram
	* The minimal code to reproduce the model training has been exported from the notebooks to /training/bigram

3. Language Modelling: Part 2
  * Description:
  	* We expand the previous character-level language model, by using more characters and by implementing a multilayer perceptron (MLP).
  * Links:
    * [Youtube-Lecture](https://www.youtube.com/watch?v=TCH_1BHY58I)
    * [Karpathys-Github](https://github.com/karpathy/makemore)
  * Files:
  	* A walkthrough of all the information given in the lecture has been written to notebook two in the folder /notebooks/bigram

4. Language Modelling: Part 3
  * Description:
  	* We examine and diagnose the multilayer perceptron (MLP) and learn about the batch normalization layer.
  * Links:
    * [Youtube-Lecture](https://www.youtube.com/watch?v=P6sfmUTpUmc)
    * [Karpathys-Github](https://github.com/karpathy/makemore)
  * Files:
  	* A walkthrough of all the information given in the lecture has been written to notebook three in the folder /notebooks/bigram

5. Language Modelling: Part 4
  * Description:
  	* We dive deep into backpropagation by manually going backwards through the 2-layer MLP (with BatchNorm) from the previous lesson. Along the way we learn to understand how pytorch is optimised, and we learn how to debug the networks better.
  * Links:
    * [Youtube-Lecture](https://www.youtube.com/watch?v=q8SA3rM6ckI)
    * [Karpathys-Github](https://github.com/karpathy/makemore)
  * Files:
  	* A walkthrough of all the information given in the lecture has been written to notebook four in the folder /notebooks/bigram

6. Language Modelling: Part 5
  * Description:
  	* We will be creating a convolutional neural network with a similar architecture to that of WaveNet (2016) from DeepMind. 
  * Links:
    * [Youtube-Lecture](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
    * [Karpathys-Github](https://github.com/karpathy/makemore)
  * Files:
  	* A walkthrough of all the information given in the lecture has been written to notebook five and six in the folder /notebooks/bigram

7. GPT From Scratch
  * Description:
  	* We will be building a Generatively Pretrained Transformer (GPT) from scratch by following the "Attention is All You Need" paper. 
  * Links:
    * [Youtube-Lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY)
    * [Karpathys-Github](https://github.com/karpathy/nanoGPT)
  * Files:
  	* A walkthrough of all the information given in the lecture has been written to notebook one in the folder /notebooks/gpt

