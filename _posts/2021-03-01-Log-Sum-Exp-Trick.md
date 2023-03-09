---
title: The Log-Sum-Exp Trick
tags: [""]
created: 2023-03-09 14:47
toc: true
layout: post
description: Normalizing vectors of log probabilities is a common task in statistical modeling, but it can result in under- or overflow when exponentiating large values. The log-sum-exp trick for resolving this issue.
hide: false
---
---
# The Log-Sum-Exp Trick

### Problem Setting
When I was switching deep learning frameworks form Tensorflow to Pytorch, I noticed something interesting when building classification models; the sigmoid (for binary classification) or softmax (for multiclass classification) in the last layer of the neural network was *not* applied in the `forward()` method. In this post, we will understand why that is the case. 

In statistical modeling and machine learning, work in a logarithmic scale is often preferred. For example, when $x$ and $y$ are small numbers, multiplying them together can cause underflow. However, if we convert to a logarithmic scale, we can convert multiplication to addition:
$$
log(xy) = log(x) + log(y)
\tag{1}
$$
This is just one reason that working with quantities such as log likelihoods and log probabilities is often preferred. For a more detailed example, consider computing a matrix determinant. This is a routine computation in many standard libraries like `SciPy`. To compute the determinant of matrix $\sum$ , these libraries use the fact that for a $D$x$D$ matrix $M$ with eigenvalues $\lambda_{1,}..., \lambda_{D}$, the determinant is equal to the product of the eigenvalues or:
$$
det(M) = \prod_{d=1}^D\lambda_d
\tag{2}
$$
However, computing the determinant this way can be numerically unstable, since if $\lambda_n$ is small, the computed determinant might be rounded to 0 due to our computer's floating point precision. And taking the log of 0 will result in `-inf`. 

Lets do this in code:
```python
>>>import numpy as np
>>>A = np.ones(100) * 1e-5
>>>np.linalg.det(np.diag(A))
0.0
```

### Floating Point Precision with Log Likelihoods
Recalling equation 2, where the determinant of a diagonal matrix M is the product of the elements along its diagonal, we can take the log of this, resulting in the following:

$$
log(det(M)) = log(\prod_{d=1}^D\lambda_{d})
\tag{3}
$$
which is equal to 
$$
= \sum_{d}^{D} log(\lambda_i)
\tag{4}
$$
If we compute equation 4 instead of 3, we might avoid an issue with floating point precision because we’re taking the log of much bigger numbers and then adding them together. Lets take a look in python:
```python
>>>A = np.ones(100) * 1e-5
>>>np.linalg.det(np.diag(A))
0.0
>>>np.log(A).sum()
-1151.2925464970228
```

We can check ourselves to see if our calculations give us the same number:
```python
>>>A = np.ones(100) * 2
>>>np.log(np.linalg.det(np.diag(A)))
69.31471805599459
>>>np.log(A).sum()
69.31471805599453
```

### The Solution
With the above concepts in mind, consider the log-sum-exp operation:
$$
LSE(x_1,...,x_{N)}= log(\sum_{n=1}^Nexp(x_n))
\tag{5}
$$


Consider the softmax activation function:
$$
p_{i}= \frac{exp(x_i)}{\sum\limits_{n=1}^{N}exp(x_{n})}
\tag{6}
$$
where $\sum\limits_{n=1}^{N} p_{n}= 1$. 
Since each $x_n$ is a log probability that might be very large and either negative or positive, then exponentiating might result in under or overflow. Therefore, we will seek to rewrite the denominator to avoid this issue. First, lets rewrite the summation term in equation 6 as
$$
exp(x_{i}) = p_i\sum_{n=1}^Nexp(x_n)
\tag{7}
$$
$$
x_{i}= log(p_{i})+ log(\sum_{n=1}^{N}exp(x_n))
\tag{8}
$$
$$
log(p_{i})= x_{i}- log(\sum_{n=1}^{N}exp(x_n))
\tag{9}
$$
$$
p_i= exp(x_{i}- \underset{LSE}{log(\sum_{n=1}^{N}exp(x_n))})


\tag{10}
$$
We can see that we have the LSE from equation 5. So again, what we have done is perform the normalization in (6) using the log-sum-exp in (5). What is nice about (5) which we did not mention is that it can be shown to equal: 
$$
y=log(\sum_{n=1}^Nexp(x_n))
\tag{5}
$$
$$
e^{y}= \sum_{n=1}^Nexp(x_n)
$$
$$
e^{y} = e^c\sum_{n=1}^Nexp(x_n-c)
$$
$$
y=c+log\sum_{n=1}^{N}exp(x_n-c)
\tag{11}
$$
This means, you can shift the center of the exponential sum by an arbitrary constant c while still computing the same final value. Critically, we’ve been able to create a term that doesn’t involve a log or exp function. Now all that’s left is to pick a good value for c that works in all cases. It turns out $c = max(x_{1}, ..., x_{N})$ works really well.

Lets try it out! Here is an example of how we can use the log-sum-exp to deal with a case of overflow:

```python
>>>x=np.array([1000, 1000, 1000])
>>>np.exp(x)
array([inf, inf, inf])
```

Now for the log-sum-exp:
```python
def logsumexp(x):
	c = x.max()
	return c + np.log(np.sum(np.exp(x-c)))

>>>logsumexp(x)
1001.0986122886682
>>>np.exp(x-logsumexp(x))
array([0.33333333, 0.33333333, 0.33333333])
```

