toc: true
layout: post
description: A minimal example of using markdown with fastpages.
categories: [markdown]
title: An Example Markdown Post
---
# Pretraining Persian AWD-LSTM Language model
> An overview of pretraining language model and use it for classification with the ULMFIT approach in Persian language model

In this post, I want to show an overview of pretraining AWD-LSTM language model and use it for classification with the ULMFIT approach. The AWD-LSTM was first introduced in the paper [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182v1.pdf). as authors stated in the paper:

 `ASGD Weight-Dropped LSTM, or AWD-LSTM, is a type of recurrent neural network that employs DropConnect for regularization, as well as NT-ASGD for optimization - non-monotonically triggered averaged SGD - which returns an average of last iterations of weights. Additional regularization techniques employed include variable length backpropagation sequences, variational dropout, embedding dropout, weight tying, independent embedding/hidden size, activation regularization and temporal activation regularization.`

For more information about the details and architecture of this model, you can refer to this post: [Understanding building blocks of ULMFIT](https://blog.mlreview.com/understanding-building-blocks-of-ulmfit-818d3775325b)
Also, here is the implementation of the model in the fastai library:[AWD_LSTM](https://docs.fast.ai/text.models.awdlstm.html#AWD_LSTM)

So here is the overview of pretraining the AWD-LSTM language model in Persian:

**Data**

This data was a subset larger data set crawled from various blog posts, news articles, and Wikipedia articles. I collected and normalized ~188k of articles for this model that can make our model more general.

Soon I'll publish the dataset in [huggingface hub](https://huggingface.co/datasets) so that it will be available for further experiment.

**Tokenizer**

Instead of fastai default tokenizer, which is Spacy, I chose SentencePiece tokenizer. The main reason behind this choice was we have some prefixes and words which decrease the language model performance after tokenization with Spacy tokenizer, but SentencePiece tokenizer fills this gap by implementing subword units.

Note that I used built-in SentencePiece tokenizer of fastai.

Here is some additional information about different kinds of tokenizers::

[Summary of the tokenizers](https://huggingface.co/transformers/tokenizer_summary.html)

[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf)

**Training**

Fastai offers a bunch of handy tools when it comes to training like [1cycle policy](https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy), [Learning Rate Finder](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) and so on. you can checkout the training script here [train](https://github.com/saied71/Persian-ULMFIT/blob/main/train.py)

I've used P3 instance of AWS for training, which has an NVIDIA V100 GPU, and it took almost 19 hours to train for 10 epochs.

**Evaluation and Model description**

Here are the metrics for the last epoch:

|epoch|train_loss|valid_loss|accuracy|perplexity|
|---|---|---|---|---|
|9|3.87870|3.90528|0.3129|49.66|

Also I set vocab-size 30000 for tokenizer.

You can follow up fine-tunning this model in this post:

[Finetuning Language model Using ULMFIT Approach in Persian language](https://saied71.github.io/tazarvAiLab/2021/07/15/Finetunin-Persian-Language-Model.html)

**Future Works**

The next step will be pretraining the same model architecture for the Estonian language. Stay tunned!!!