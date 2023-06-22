---
title: PEFT with LoRA and QLoRA
image: images/LoRA.png
layout: post
created: 2023-06-22
description: Large Language Model (LLM) fine-tuning with low-rank matrix
  approximations and data-type quantization.
tag: LLMs, LoRA, PEFT, QLoRA, finished_post
---
# Introduction
Since the inception of transfer learning, dozens of works have sought to make model adaptation more parameter- and compute-efficient. Yet it is still the case that one of main drawbacks for LLMs is that have to be fine-tuned for *each* downstream task, learning a different set of parameters. 

> [!Question]
> Why can we use relatively vanilla gradient descent algorithms (e.g., without strong regularization) to tune a model with hundreds of millions of parameters on datasets with only hundreds or thousands of labeled examples?


# PEFT: Parameter Efficient Fine Tuning

## LoRA
**Low-Rank Adaptation, or LoRA**[^1]- which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. The most significant benefit comes from the reduction in memory and storage usage.

### Refresher on Matrix Decomposition
SVD representation of a matrix is the singular value decomposition of any matrix A so that $A=USV^T$. 
- $U$ is the left singular vectors obtained by finding an orthonormal set of eigenvectors $A^TA$. 
- $S$ is a diagonal matrix of singular values, which are the square roots of the eigenvalues of $AA^T$
- V is the right singular vectors obtained by finding an orthonormal set of eigenvectors t$A^TA$

You can truncate the SVD of a higher-rank matrix $A$ to get the a low-rank approximation. This is done by setting all but the first $k$ largest singular values to zero, and using the first $k$ rows and columns of $U$ and $V$. The rank is the $k$ chosen. This works because singular values decrease exponentially with rank, with earlier singular values being much larger than later ones. 

### Low-Rank Parameterized Update Matrices
A neural network contains many dense layers which perform matrix multiplication. The weight matrices in these layers typically have full-rank. When adapting to a specific task, Aghajanyan et al. (2020)[^2] shows that the pre-trained language models have a low “instrisic dimension” and can still learn efficiently despite a random projection to a smaller subspace. In other words, there exists a low dimension reparameterization that is as effective for fine-tuning as the full parameter space.

> [!note]
> An objective function’s intrinsic dimensionality describes the minimum dimension needed to solve the optimization problem it defines to some precision level. In the context of pretrained language models, measuring intrinsic dimensional will tell us how many free parameters are required to closely approximate the optimization problem that is solved while fine-tuning for each end task.

Inspired by this, the LoRA authors hypothesize the updates to the weights also have a low “intrinsic rank” during adaptation.

For a pretrained weight matrix $W_{0}\in \mathbb{R}^{d \times k}$, the update is constrained by representing its with a low-rank decomposition $W_{0}+ \bigtriangleup W = W_{0}+ BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and the rank $r << min(d, k)$. During training, $W_0$ is frozen and does not receive gradient updates, while $A$ and $B$ are trainable parameters. So the forward pass yields:

$$
h = W_{0x}+ \bigtriangledown Wx = W_{0x}+ BAx
$$

In other words, gradients during stochastic gradient descent are passed through the fixed pretrained model weights to the adapter, which is updated to optimize the loss function. LoRA augments a linear projection through an additional factorized projection.

Using the technique shown above, **_r(n + k)_** parameters have to be tuned during model adaption. Since **_r << min(n, k)_**, this is much lesser than the number of parameters that would have to be tuned otherwise (**_nk_**). This reduces the time and space required to finetune the model by a large margin. Some numbers from the paper and our experiments are discussed in the sections below.

To take an extreme example, supposed the $W_O$ is of size 512x512. That is 512$^2$ parameters. On the other hand, using two matrices via LoRA to replace $W_O$ where $L_{1}\in \mathbb{R}^{512 \times 1}$ and $L_{2}\in \mathbb{R}^{1 \times 512}$, that is only 1024 total parameters. 

### Simple LoRA Implementation
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-3b", 
    torch_dtype=torch.float16,
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")

print(model)

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Obtain LoRA Model
from peft import LoraConfig, get_peft_model 

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Load sample data
from datasets import load_dataset

qa_dataset = load_dataset("squad_v2")

# Train LoRA
import transformers

trainer = transformers.Trainer(
    model=model, 
    train_dataset=mapped_qa_dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=100,
        learning_rate=1e-3, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
```
[^4]

### Where LoRA Falls Short
While LoRA was designed as a PEFT method, most of the memory footprint for LLM finetuning comes from activation gradients and not from the learned LoRA parameters. For example, for a 7B LLaMA model trained on FLAN v2 with a batch size of 1, with LoRA weights equivalent to commonly used 0.2% of the original model weights, the LoRA input gradients have a memory footprint of 567 MB while the LoRA parameters take up only 26 MB.

## QLoRA- Efficient Finetuning of Quantized LLMs
QLoRA reduces the average memory needs of finetuning a 65B parameter model from >780GB of GPU RAM to 48GB without sacrificing runtime or predictive performance.[^3] This is done via an algorithm developed by researchers at the University of Washington that quantizes a pretrained model using to a 4-bit resolution before adding a sparse set of learnable Low-rank Adapter weights modified by backpropagating gradients through the quantized consequences. As a result, QLoRA  has made the largest publicly available models to date fine-tunable on a single GPU. 

#### What is Quantization?
Quantization is the process of discretizing an input from a representation that holds more information to a representation with less information. It often means taking a data type with more bits and converting it to fewer bits, for example from 32-bit floats to 8-bit Integers. To ensure that the entire range of the low-bit data type is used, the input data type is commonly rescaled into the target data type range through normalization by the absolute maximum of the input elements, which are usually structured as a tensor. 

> This is important since to calculate the model size in bytes, one multiplies the number of parameters by the size of the chosen precision in bytes.

**Example 1**
Let say we want to go from `float32` to `int8`.
```
array = [-1024, 5, 2048, 256] # as type float32
```

To do the conversion, we will multiply each element in the array by a quantization factor `c`. To calculate `c` , we first find the max bit number for the desired conversion type (so in this case for `int8` it is 127 since the range of `int8` is `[-127, 127]`), and then divide it by the max number of the array. 

Then, for each item in our array, we divide the item by the quantization factor and round, thus achieving quantization albeit with an expected loss of precision. 

```
>>> c = 127/max(array)
>>> c
0.62

>>> quantized = [round(i/c) for i in array]
>>> quantized
[-64, 0, 127, 16]
```

**Example 2** - why is this problematic
Using the same conversion instead for `[0, 100, 100000]`, you will get 0 for the first two elements of the array, leading to a lot of information loss. The problem with this approach is that if a large magnitude value (i.e., an outlier) occurs in the input tensor, then the quantization bins—certain bit combinations—are not utilized well with few or no numbers quantized in some bins


### QLoRA's Solution
QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) double quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) paged optimizers to manage memory spikes.
 
4-bit NormalFloat quantization improves upon quantile quantization by ensuring an equal number of values in each quantization bin. This avoids computational issues and errors for outlier values. During finetuning, QLoRA backpropagates gradients through the frozen 4-bit quantized pretrained language model into the Low-Rank Adapters. QLoRA dequantizes weights from the 4-bit NormalFloat (what the authors call storage data type) to the 16-bit BrainFloat (computation data type) to perform the forward and backward passes, but only computes weight gradients for the LoRA parameters which use 16-bit BrainFloat. The weights are decompressed only when they are needed, therefore the memory usage stays low during training and inference.

[^1]: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, & Weizhu Chen (2021). LoRA: Low-Rank Adaptation of Large Language Models_. CoRR, _abs/2106.09685_. <https://arxiv.org/pdf/2106.09685.pdf>
[^2]: Armen Aghajanyan, Luke Zettlemoyer, & Sonal Gupta (2020). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning_. CoRR, _abs/2012.13255_, <https://arxiv.org/abs/2012.13255.>
[^3]: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, & Luke Zettlemoyer. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
[^4]: https://colab.research.google.com/drive/1iERDk94Jp0UErsPf7vXyPKeiM4ZJUQ-a?usp=sharing#scrollTo=kfAO01v-qEPS