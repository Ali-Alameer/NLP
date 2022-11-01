# NLP
This repository provides methods and tutorials of Natural Language Processing (NLP), manily based on keras/tensorflow. These will be discussed in detail as part of the NLP module of the MSc in Artificial Intelligence programme at the Univesity of Salford. 

{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Representing text\n",
        "\n",
        "If we want to solve Natural Language Processing (NLP) tasks with neural networks, we need some way to represent text as tensors. Computers already represent characters as numbers that map to letters on your screen using encodings such as ASCII or UTF-8.\n",
        "\n",
        "![Image showing diagram mapping a character to an ASCII and binary representation](https://learn.microsoft.com/en-gb/training/modules/intro-natural-language-processing-tensorflow/notebooks/images/ascii-character-map.png)\n",
        "\n",
        "We understand what each letter **represents**, and how all characters come together to form the words of a sentence. However, computers don't have such an understanding, and neural networks have to learn the meaning of the sentence during training.\n",
        "\n",
        "We can use different approaches when representing text:\n",
        "* **Character-level representation**, where we represent text by treating each character as a number. Given that we have $C$ different characters in our text corpus, the word *Hello* could be represented by a tensor with shape $C \\times 5$. Each letter would correspond to a tensor in one-hot encoding.\n",
        "* **Word-level representation**, in which we create a **vocabulary** of all words in our text, and then represent words using one-hot encoding. This approach is better than character-level representation because each letter by itself does not have much meaning. By using higher-level semantic concepts - words - we simplify the task for the neural network. However, given a large dictionary size, we need to deal with high-dimensional sparse tensors.\n",
        "\n",
        "Let's start by installing some required Python packages we'll use in this module."
      ]
    },
