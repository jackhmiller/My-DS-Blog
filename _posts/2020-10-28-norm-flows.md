---
title: Probabilistic Generative Models- Normalizing Flows
image: images/norm_flow.png
layout: post
created: 2023-07-11
description: ""
tag: generative_model, normalizing_flows
toc: "true"
---

## Introduction
Both GAN and VAE lack the exact evaluation and inference of the probability distribution, which often results in low-quality blur results in VAEs and challenging GAN training in GANs with challenges such as mode collapse and vanishing gradients posterior collapse, etc. VAEs specifically can sometimes suffer from blurry or incomplete reconstructions due to the limitations of the variational inference framework. This is because due to the complexity of real-world data distributions, the mapping from the latent space to the data space may not cover the entire data space, leading to incomplete coverage and potentially missing regions of the target distribution. 

Normalizing flows were proposed to solve many of the current issues with GANs and VAEs by using reversible functions to explicitly model the mapping from the latent space to the data space without relying on an approximation. The main idea is to learn a mapping from a simple distribution to a target distribution by composing a sequence of invertible transformations, where each transformation is itself an invertible neural network. By applying these transformations, the model can learn to capture the complex dependencies and multimodal nature of the target distribution.

## Mathematical Prerequisites
Change of variables:

$$
p_x(x) = p_z(f(x))|\text{det}Df(x)|
\tag{1}
$$

where:
- $p_x(x)$ is the density of $x$
- $p_z(x)$ is the density of $z$$ - our **latent distribution**
- $f(x)$ is an invertible, differentiable transformation 
- $|\text{det}Df(x)|$ is a volume correction term that ensures that $x$ and $z$ are valid probability measures (integrate to 1)[^3]
- $Df(x)$ is the Jacobian of $f(x)$

Ex.
![]({{ site.baseurl }}/images/NF_change_of_variables.png "Source: Badly drawn by the author")

Ultimately, we want to find a $p_z(x)$ that is easy to work with, by finding the function that transforms the complex distribution $p_x(x)$ (our data) into $p_z(x)$.  So for example, making the distribution of our images $p_x(x)$ look like Gaussian noise. 

## Where Normalizing Flows Comes In
Normalizing flows says that given eq. 1, we want to find the $f(x)$, which is our *flow*.  Our base measure $p_z(x)$ is typically selected as $N(z|0, I)$, hence the *normalizing* in normalizing flows. So if we can find $f(x)$ that gets us from $p_x(x)$ to $p_z(x)$, then we can perform the two tasks vital for PGMs:
1. **Sample**- $z \sim p_z(\cdot)$
2. **Evaluate\compute** $x = f^{-1}(z)$

We train the NF by maximum (log-) likelihood:

$$
\underset{\theta}{argmin}\sum_{i=1}^N\text{log}p_z(f(x_i|\theta)) + \text{log}|\text{det}Df(x_i|\theta)|
\tag{2}
$$

where $\theta$ are the parameters of the flow $f(x|\theta)$. It is constructing these flows that is the key research problem with NFs. 

![]({{ site.baseurl }}/images/NF_sketch.png "Source: Justin Solomon,  MIT 6.838: Shape Analysis")

In terms of notation in the above sketch, we want to learn the parameters of $T_\phi$ that achieve the bijective mapping. 

> TLDR: Find the parameters of the normalizing flow that best explains our data, by maximizing the likelihood of observing our data given those parameters 

This can also be thought of as minimizing the KL divergence between $p_x(x)$ and $p_z(x)$. Similar to our approach in VAE, we can evaluate $KL((p_z(x)||(p_x(x)))$ as $=\text{const.} - \mathbb{E}_{x\sim p_z(x)}[\text{log}p_x(x)]$. If we replace $p_x(x)$ in the expectation, we get eq. 2 above!

This is the advantage with normalizing flows:

>[!Takeaway]
>We can sample from the base measure ($z$), push it through the transform (flows) $f$, and then minimize the KL divergence like we do with VAEs but it a more differentiable way. 

## More on the Flows
Instead of a single flow $f(x)$, we are going to use a composition of flows. We can do this since invertible, differentiable functions are closed under composition[^2]:

$$
f = f_K\circ f_{K-1}\circ...\circ f_2 \circ f_1
$$

This allows us to build a complex flow from the composition of simpler flows. This is similar to stacking layers in a DNN. 

![]({{ site.baseurl }}/images/NF_flows.png "Source: [^1]")

This means that the determinant term in eq. 1 and eq. 2 can now be expressed as the product of the determinants of the individual functions:

$$
\text{det}Df = \text{det}\prod Df = \prod_{k=1}^K \text{det} Df_k
$$

So now we can rewrite eq. 2 as:

$$
\underset{\theta}{max}\sum_{i=1}^N\text{log}p_z(f(x_i|\theta)) + \sum_{k=1}^K\text{log}|\text{det}Df_k(x_i|\theta)|
\tag{3}
$$

## Coupling Flows
We use **coupling flows** as a general approach to construct non-linear flows. We partition the parameters into two disjoint subsets $x = (x^A, x^B)$ then 

$$
f(x) = (x^A, \hat{f}(x^B|\theta(x^A)))
$$

where $\hat{f}(x^B|\theta(x^A))$ is another flow but whose parameters depend on $x^A$. 

**<u>Forward</u>- the flow direction**
![]({{ site.baseurl }}/images/coupling_flow_forward.png "Source: Marcus Brubaker, A tutorial on Normalizing Flows")


The $\theta$ in the coupling network above can be an arbitrarily complex function, like a MLP, CNN, etc. The parameters output from the coupling network are then applied to the second split in what is labeled as the coupling transform above.

For example, if $f$ is a linear transformation (an affine flow), the coupling network learns $s$ and $t$ to represent scale and translation using $x^A$. Then $f$ applies to $x^B$ using $s, t$ would look like $s \cdot x^B + t$.   

We repeat this splitting process, applying shuffling, and go around and around until each part of $x$ has a chance to be a part of the conditioning coupling network. 

**<u>Inverse:</u> the generative direction**
![]({{ site.baseurl }}/images/coupling_flow_reverse.png "Source: Marcus Brubaker, A tutorial on Normalizing Flows")


## Training Process
1. Get a example from the dataset $x$
2. Flow to $z$ space $z=f_\theta(x)$
3. Account for change in probability mass by calculating the determinant of the Jacobian $\text{log}|\text{det}Df_\theta(x)|$ which for the affine flow is just the determine of the scale $\sum\text{log}|s|$
4. Calculate the loss: $L = |z|_2^2 + \sum\text{log}|z|$
5. Update $\theta$

![]({{ site.baseurl }}/images/NF_training_example.png "Source: Hans van Gorp- https://www.youtube.com/watch?v=yxVcnuRrKqQ")


[^1]: https://mbrubake.github.io/eccv2020-nf_in_cv-tutorial/
[^2]: Function composition is a way of combining two functions to create a new function by applying one function to the output of the other function. A set or collection of objects is said to be closed under a certain operation if applying that operation to any two objects in the set always produces another object that is also in the set. For example, consider a set of functions S. We say that this set of functions is closed under composition if, for any two functions f and g in S, the composition of f and g, denoted as f(g(x)), is also in S.
[^3]: The determinant of the Jacobian matrix encapsulates how a function distorts or stretches the local space around a point. It quantifies the local scaling factor and provides information about orientation preservation or reversal. If it is large, it means that the transformation significantly stretches or shrinks the local space, given a measure of local scaling. The sign of the determinant tells us whether the transformation preserves or reverses the orientation of the local space.
