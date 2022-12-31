CUDA out of memory is one of the most common and also painful errors while training deep learning models. One of the most straightforward solutions is to reduce the batch size but that may not work if you are dealing with large models and then WHAT?!!

Here I discuss approaches to overcome this pain. Keep in mind that this short blog is heavily borrowed from the HuggigeFace blog so if you want to dive deep into these topics check out [Efficient Training on a Single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one) 




https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one

## Gradient Accumulation
Gradient accumulation is a technique where you can train on larger batch sizes than your machine would normally be able to fit into memory. The way we do that is to calculate the gradients iteratively in smaller batches by doing a forward and backward pass through the model and accumulating the gradients in the process. When enough gradients are accumulated we run the model’s optimization step. This way we can easily increase the overall batch size to numbers that would never fit into the GPU’s memory. Adding forward and backward passes, however, may slow down training.

You can pass `gradient_accumulation_steps` to transformers' `TrainingArguments` to specify steps to accumulate the gradients before performing a backward/update pass.
```python
training_args = TrainingArguments(
    output_dir="out",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    report_to="none",
    gradient_accumulation_steps=8,
)
```

## Gradient Checkpointing
Even when we set the batch size to 1 and use gradient accumulation we can still run out of memory when working with large models. In order to compute the gradients during the backward pass all activations from the forward pass are normally saved. This can create a big memory overhead. Alternatively, one could forget all activations during the forward pass and recompute them on demand during the backward pass.

Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, the checkpointed part does not save intermediate activations, and instead recomputes them in backward pass. 
We can pass `gradient_checkpointing` argument to `TrainingArguments` :
```python
training_args = TrainingArguments(
    output_dir="out",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    report_to="none",
    gradient_accumulation_steps=8,
    gradient_checkpointing=True
)
```
Just kepp in mind that gradient checkpointing would add a significant computational overhead and slow down training. 

To overcome slow training you can `half precision` or `fp16` training by passing `fp16` argument to `TrainingArguments` 
```python
training_args = TrainingArguments(
    output_dir="out",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    report_to="none",
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    fp16=True
)
```
Just keep in mind to use `fp16` you need to install Nvidia [apex](https://github.com/NVIDIA/apex) by running this bash code:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

To understand the detail of `fp16` you can checkout this blog : [Accelerating AI Training with NVIDIA TF32 Tensor Cores](https://github.com/NVIDIA/apex)

## Optimizer
The most common optimizer used to train transformer model is [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/#:~:text=Adam%20is%20a%20replacement%20optimization,sparse%20gradients%20on%20noisy%20problems.) or [AdamW](https://towardsdatascience.com/why-adamw-matters-736223f31b5d) (Adam with weight decay). Adam achieves good convergence by storing the rolling average of the previous gradients which, however, adds an additional memory footprint of the order of the number of model parameters. One remedy to this is to use an alternative optimizer such as Adafactor, which works well for some models but often it has instability issues. 

## Personal Experience

If you finetune large models, you may not see OOM errors at first, but in the middle of training, such as at the beginning of epoch 1. By doing some experiments I findout using `gradient_checkpointing` is the way to overcome OOM error uring the training. Keep in mind that as mentioned before using `gradient_checkpointing` comes with computation cost AKA longer training time.
## Reference:
- [Fitting larger networks into memory](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)

- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174v2)