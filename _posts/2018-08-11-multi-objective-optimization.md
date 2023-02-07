---
title: Multi-Objective Optimization
tags: ["multi-objective-optimization"]
created: 2023-02-07 09:35
toc: true
layout: post
description: A sophisticated yet simple solution to combining multiple loss functions for optimizing a single model
hide: false
---
---
# Multi-Objective Optimization
<br>

### Introduction
Machine learning is inherently a multi-objective task. In most real-world problems, a learner is required to find a model with strong performance across multiple objectives $L_1, ..., L_p$. Committing to a single objective $L_k$ fails to capture the full complexity of the underling problem and causes models to overfit to that individual objective.

### Why not use more than one loss function?
A practitioner might want to use a variety of common training loss functions, each having its own advantages and drawbacks. For example, the Zero-One loss function often results in robust models when the data contains outliers, but it underperforms when the data is concentrated near decision surface. The opposite usually applies to the hinge loss function. To solve a complex machine learning problem, the learner would wish to combine multiple loss functions, making use of their strengths and mitigating weaknesses.

Unfortunately, in multi-objective optimization, there generally exists no singular solution. Thus when considering an objective function $f$, there exists on single optimal model but rather a set of optimal models.[^1] A model $x\epsilon \chi$ dominates another model $x' \epsilon \chi$ if $f_i(x) \leq f_i(x')$ for all $i$ and there exists some $i$ such that $f_i(x) \leq f_i(x')$.

The set of all non-dominated solutions (i.e. models) is denoted as the Pareto front $\mathcal{P}_f(\chi)$ whose members are commonly referred to as the Pareto-optimal solutions:

$$
\mathcal{P}_f(\chi) = \left \{ x \in  \chi | \hspace{.2cm} \exists \hspace{.2cm} x' \in \chi: x' \prec _f x  \right \}
$$

<br>

### The inherent difficulty in multi-objective optimization 
Despite the multi-objective nature of machine learning, working with a vector of multiple objectives at the same time turns out to be computationally challenging. For that reason, researchers often combine the set of base objectives $L_1, ..., L_p$ into a weighted ensemble of the form $L_k = \sum_{k=1}^p \lambda_l L_k$, with each objective $L_k$ weighted by $\lambda_k$.  This allows one to make use of efficient scalar function optimization algorithms on large-scale problems, typically using various stochastic gradient descent techniques. However, working with an ensemble of base objectives raises a natural question: how should we set the mixture weights $\lambda_1,...,\lambda_p$?

Despite the simplicity of the questions above, there is no clear answer to how to determine the mixture weights for multi-objective problems. Particularly, this is because there is no straightforward way to map the requirements of a particular problem that a learner is seeking to solve to a corresponding distribution of the mixture weight. Thus, the mixture weights are usually assigned to uniform by default. However, in many cases the uniform ensemble of objectives can do more harm than good. This is simply because fitting models with the uniform mixture weights can significantly hurt some of the individual objectives in the mixture. For many machine learning applications (e.g., vision or speech), a significant decrease in performance on one objective is intolerable even if the performance on the uniform average of objectives is improved.

One can argue that if the uniform combination of objectives is not the natural target, then we should just set mixture weights for every problem separately based on the relative importance of each of the base losses. However, this is still not satisfactory, because the learner’s preferences are often shifting over time, they may be unobservable in some cases or even unknown. It is also often the case in the machine learning industry that when several parties develop a model for a particular multi-objective problem, their preferences for the base objectives are conflicting with each other.
<br>

#### A solution to weight assignment 
We assume that the true mixture mixture of coefficients $\lambda$, or our weights for the different loss functions, are unknown. Our goal is to maximize the weight of a respective loss function $h$ within the overall weighting schema for all of the loss functions. Thus we can define our multi-objective loss function where each loss function $h$ is parameterized by a $\lambda$. Thus, we want to minimize each loss function, while maximizing each's $\lambda$ parameter with respect to the other $\lambda s$.

With our new loss function, we can formulize our new optimization problem as
$$
\underset{h \in H}{min}L_{\Lambda}(h)
$$
Given that the hypothesis is parameterized by a vector $w \in W$, our final optimization problem can be formulated as follows:
$$
\underset{w\in W}{min} \ \underset{\lambda\in \Lambda }{max} L(w, \lambda)
= \underset{w\in W}{min} \ \underset{\lambda\in \Lambda }{max}\sum_{k=1}^p\lambda_k\frac{1}{m}\sum_{i=1}^{m}h_k(x_i, y_i)
$$
where
$$
\frac{1}{m}\sum_{i=1}^{m}h_k(x_i, y_i) = L_k(h_w)
$$
So to simplify:
$$
\underset{w\in W}{min} \ \underset{\lambda\in \Lambda }{max}\sum_{k=1}^p\lambda_k L_k(h_w)
$$
Fortunately, the optimization problem above is convex, and can be solved using gradient-based algorithms. We can also introduce regularization parameters and norms in order to control for the complexity of the hypothesis space.

#### Implementation
The optimization of our problem above can be solved using projected gradient descent, where optimization is performed over $(w, \lambda)$. To simplify our notation:
$$
L(w) = \frac{1}{m}\sum_{i=1}^{m}h_k(x_i, y_i)
$$
$$
L(w, \lambda) = \sum_{k=1}^p\lambda_k L_k(w)
$$
Or simply written:
$$
\underset{w\in W}{min} \ \underset{\lambda\in \Lambda }{max} \ L(w, \lambda)
$$
The inner maximization choses a value for $\lambda \in \Lambda$ to maximize the objective function, while the second minimization seeks to minimize loss by choosing $w \in W$.  Each gradient descent iteration will operate on the loss function with respect to $w$ as well as $\lambda$. So for each iteration $t \in [1, T]$, the updates for $w$ and $\lambda$ are as follows:

$$
w_t \leftarrow \prod W [w_{t-1} - \gamma _w\delta _wL(w_{t-1}, \lambda_{t-1})]
$$
$$
\lambda_t \leftarrow \prod \Lambda [\lambda_{t-1} - \gamma _\lambda\delta _\lambda L(w_{t-1}, \lambda_{t-1})]
$$

We can use these update rules to combined any number of loss functions into a signal objective function, that can in turn be minimized using gradient descent. 

### Experiments
Such a framework was implemented in [^2] suggest that this framework for multi-objective optimization improves the model for the worst performing loss. In certain cases it can lead to an improvement in performance, but it is better suited to increase model robustness and guard against overfitting. 

[^1]: Michael TM Emmerich and André H Deutz. A Tutorial on Multiobjective Optimization: Fundamentals and Evolutionary Methods. _Natural Computing_, 17(3):585–609, 2018.
[^2]: Cortes, C., Mohri, M., Gonzalvo, J., & Storcheus, D. (2020). Agnostic Learning with Multiple Objectives. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, & H. Lin (Eds.), _Advances in Neural Information Processing Systems_ (Vol. 33, pp. 20485–20495). Retrieved from https://proceedings.neurips.cc/paper/2020/file/ebea2325dc670423afe9a1f4d9d1aef5-Paper.pdf