---
layout: post
title: Autoregressive Models
---

We begin our study into generative modeling with autoregressive models. As before, we assume we are given access to a dataset $$\mathcal{D}$$ of $$n$$-dimensional datapoints $$\mathbf{x}$$. For simplicity, we assume the datapoints are binary, i.e., $$\mathbf{x} \in \{0,1\}^n$$.

Representation
==============

By the chain rule of probability, we can factorize the joint distribution over the $$n$$-dimensions as 

{% math %}
p(\mathbf{x}) = \prod\limits_{i=1}^{n}p(x_i \vert x_1, x_2, \ldots, x_{i-1}) = 
\prod\limits_{i=1}^{n} p(x_i \vert \mathbf{x}_{< i } )
{% endmath %}

where $$\mathbf{x}_{< i}=[x_1, x_2, \ldots, x_{i-1}]$$ denotes the vector of random variables with index less than $$i$$. 

The chain rule factorization can be expressed graphically as a Bayesian network.


<figure>
<img src="autoregressive.png" alt="drawing" width="400" class="center"/>
<figcaption>
Graphical model for an autoregressive Bayesian network with no conditional independence assumptions.
 </figcaption>
</figure>

Such a Bayesian network that makes no conditional independence assumptions is said to obey the *autoregressive* property.
The term *autoregressive* originates from the literature on time-series models where observations from the previous time-steps are used to predict the value at the current time step. Here, we fix an ordering of the variables $$x_1, x_2, \ldots, x_n$$ and the distribution for the $$i$$-th random variable depends on the values of all the preceding random variables in the chosen ordering $$x_1, x_2, \ldots, x_{i-1}$$.

If we allow for every conditional $$p(x_i \vert \mathbf{x}_{< i})$$ to be specified in a tabular form, then such a representation is fully general and can represent any possible distribution over $$n$$ random variables. However, the space complexity for such a representation grows exponentially with $$n$$.

To see why, let us consider the conditional for the last dimension, given by $$p(x_n \vert \mathbf{x}_{< n})$$. In order to fully specify this conditional, we need to specify a probability distribution for each of the $$2^{n-1}$$ configurations of the variables $$x_1, x_2, \ldots, x_{n-1}$$. For any one of the $$2^{n-1}$$ possible configurations of the variables, the probabilities should sum to one. Therefore, we need only one parameter for each configuration, so the total number of parameters for specifying this conditional is given by $$2^{n-1}$$. Hence, a tabular representation for the conditionals is impractical for learning the joint distribution factorized via chain rule.

In an *autoregressive generative model*, the conditionals are specified as parameterized functions with a fixed number of parameters. That is, we assume the conditional distributions $$p(x_i \vert \mathbf{x}_{< i})$$ to correspond to a Bernoulli random variable and learn a function that maps the preceding random variables $$x_1, x_2, \ldots, x_{i-1}$$ to the
mean of this distribution. Hence, we have
{% math %}
p_{\theta_i}(x_i \vert \mathbf{x}_{< i}) = \mathrm{Bern}(f_i(x_1, x_2, \ldots, x_{i-1}))
{% endmath %}
where $$\theta_i$$ denotes the set of parameters used to specify the mean
function $$f_i: \{0,1\}^{i-1}\rightarrow [0,1]$$. 


The number of parameters of an autoregressive generative model are given by $$\sum_{i=1}^n \vert \theta_i \vert$$. As we shall see in the examples below, the number of parameters are much fewer than the tabular setting considered previously. Unlike the tabular setting however, an autoregressive generative model cannot represent all possible distributions. Its expressiveness is limited by the fact that we are limiting the conditional distributions to correspond to a Bernoulli random variable with the mean specified via a restricted class of parameterized functions.

<figure>
<img src="fvsbn.png" alt="drawing" width="200" class="center"/>
<figcaption>
 A fully visible sigmoid belief network over four variables. The conditionals are denoted by \(\widehat{x}_1, \widehat{x}_2, \widehat{x}_3, \widehat{x}_4\) respectively.
 </figcaption>
</figure>
In the simplest case, we can specify the function as a linear combination of the input elements followed by a sigmoid non-linearity (to restrict the output to lie between 0 and 1). This gives us the formulation of a *fully-visible sigmoid belief network* ([FVSBN](https://papers.nips.cc/paper/1153-does-the-wake-sleep-algorithm-produce-good-density-estimators.pdf)).

{% math %}
f_i(x_1, x_2, \ldots, x_{i-1}) =\sigma(\alpha^{(i)}_0 + \alpha^{(i)}_1 x_1 + \ldots + \alpha^{(i)}_{i-1} x_{i-1})
{% endmath %} 

where $$\sigma$$ denotes the sigmoid function and $$\theta_i=\{\alpha^{(i)}_0,\alpha^{(i)}_1, \ldots, \alpha^{(i)}_{i-1}\}$$ denote the parameters of the mean function. The conditional for variable $$i$$ requires $$i$$ parameters, and hence the total number of parameters in the model is given by $$\sum_{i=1}^ni= O(n^2)$$. Note that the number of parameters are much fewer than the exponential complexity of the tabular case.

A natural way to increase the expressiveness of an autoregressive generative model is to use more flexible parameterizations for the mean function e.g., multi-layer perceptrons (MLP). For example, consider the case of a neural network with 1 hidden layer. The mean function for variable $$i$$ can be expressed as

{% math %}
\mathbf{h}_i = \sigma(A_i \mathbf{x_{< i}} + \mathbf{c}_i)\\
f_i(x_1, x_2, \ldots, x_{i-1}) =\sigma(\boldsymbol{\alpha}^{(i)}\mathbf{h}_i +b_i )
{% endmath %}

where $$\mathbf{h}_i \in \mathbb{R}^d$$ denotes the hidden layer activations for the MLP and $$\theta_i = \{A_i \in \mathbb{R}^{d\times (i-1)},  \mathbf{c}_i \in \mathbb{R}^d, \boldsymbol{\alpha}^{(i)}\in \mathbb{R}^d, b_i \in \mathbb{R}\}$$
are the set of parameters for the mean function $$\mu_i(\cdot)$$. The total number of parameters in this model is dominated by the matrices $$A_i$$ and given by $$O(n^2 d)$$.


<figure>
<img src="nade.png" alt="drawing" width="200" class="center"/>
<figcaption>
 A neural autoregressive density estimator over four variables. The conditionals are denoted by \(\widehat{x}_1, \widehat{x}_2, \widehat{x}_3, \widehat{x}_4\) respectively. The blue connections denote the tied weights \(W[., i]\) used for computing the hidden layer activations.
 </figcaption>
</figure>

The *Neural Autoregressive Density Estimator* ([NADE](http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf)) provides an alternate MLP-based parameterization that is more statistically and computationally efficient than the vanilla approach. In NADE, parameters are shared across the functions used for evaluating the conditionals. In particular, the hidden layer activations are specified as

{% math %}
\mathbf{h}_i = \sigma(W_{., < i} \mathbf{x_{< i}} + \mathbf{c})\\
f_i(x_1, x_2, \ldots, x_{i-1}) =\sigma(\boldsymbol{\alpha}^{(i)}\mathbf{h}_i +b_i )
{% endmath %}
where $$\theta=\{W\in \mathbb{R}^{d\times n}, \mathbf{c} \in \mathbb{R}^d, \{\boldsymbol{\alpha}^{(i)}\in \mathbb{R}^d\}^n_{i=1}, \{b_i \in \mathbb{R}\}^n_{i=1}\}$$is
the full set of parameters for the mean functions $$f_1(\cdot), f_2(\cdot), \ldots, f_n(\cdot)$$. The weight matrix $$W$$ and the bias vector $$\mathbf{c}$$ are shared across the conditionals. Sharing parameters offers two benefits:

1.  The total number of parameters gets reduced from $$O(n^2 d)$$ to $$O(nd)$$ \[readers are encouraged to check!\].

2.  The hidden unit activations can be evaluated in $$O(nd)$$ time via the following recursive strategy:
    {% math %}
    \mathbf{h}_i = \sigma(\mathbf{a}_i)\\
    \mathbf{a}_{i+1} = \mathbf{a}_{i} + W[., i]x_i
    {% endmath %}
    with the base case given by $$\mathbf{a}_1=\mathbf{c}$$.


###  Extensions to NADE

The [RNADE](https://arxiv.org/abs/1306.0186) algorithm extends NADE to learn generative models over real-valued data. Here, the conditionals are modeled via a continuous distribution such as a equi-weighted mixture of $$K$$ Gaussians. Instead of learning a mean function, we now learn the means $$\mu_{i,1}, \mu_{i,2},\ldots, \mu_{i,K}$$ and variances $$\Sigma_{i,1}, \Sigma_{i,2},\ldots, \Sigma_{i,K}$$ of the $$K$$ Gaussians for every conditional. For statistical and computational efficiency, a single function $$g_i: \mathbb{R}^{i-1}\rightarrow\mathbb{R}^{2K}$$ outputs all the means and variances of the $$K$$ Gaussians for the $$i$$-th conditional distribution.

Notice that NADE requires specifying a single, fixed ordering of the variables. The choice of ordering can lead to different models. The [EoNADE](https://arxiv.org/abs/1310.1757) algorithm allows training an ensemble of NADE models with different orderings.

Learning and inference
======================

Recall that learning a generative model involves optimizing the closeness between the data and model distributions. One commonly used notion of closeness in the KL divergence between the data and the model distributions.

{% math %}
\min_{\theta\in \mathcal{M}}d_{KL}
(p_{\mathrm{data}}, p_{\theta}) = \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}} }\left[\log p_{\mathrm{data}}(\mathbf{x}) - \log p_{\theta}(\mathbf{x})\right]
{% endmath %}

Before moving any further, we make two comments about the KL divergence. First, we note that the KL divergence between any two distributions is asymmetric. As we navigate through this chapter, the reader is encouraged to think what could go wrong if we decided to optimize the reverse KL divergence instead. Secondly, the KL divergences heavily penalizes any model distribution $$p_\theta$$ which assigns low probability to a datapoint that is likely to be sampled under $$p_{\mathrm{data}}$$. In the extreme case, if the density $$p_\theta(\mathbf{x})$$ evaluates to zero for a datapoint sampled from $$p_{\mathrm{data}}$$, the objective evaluates to $$+\infty$$.

Since $$p_{\mathrm{data}}$$ does not depend on $$\theta$$, we can equivalently recover the optimal parameters via maximizing likelihood estimation.

{% math %}
\max_{\theta\in \mathcal{M}}\mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}} }\left[\log p_{\theta}(\mathbf{x})\right].
{% endmath %}

Here, $$\log p_{\theta}(\mathbf{x})$$ is referred to as the log-likelihood of the datapoint $$\mathbf{x}$$ with respect to the model distribution $$p_\theta$$.

To approximate the expectation over the unknown $$p_{\mathrm{data}}$$, we make an assumption: points in the dataset $$\mathcal{D}$$ are sampled i.i.d. from $$p_{\mathrm{data}}$$. This allows us to obtain an unbiased Monte Carlo estimate of the objective as

{% math %}
\max_{\theta\in \mathcal{M}}\frac{1}{\vert D \vert} \sum_{\mathbf{x} \in\mathcal{D} }\log p_{\theta}(\mathbf{x}) = \mathcal{L}(\theta \vert \mathcal{D}).
 {% endmath %}


The maximum likelihood estimation (MLE) objective has an intuitive interpretation: pick the model parameters $$\theta \in \mathcal{M}$$ that maximize the log-probability of the observed datapoints in $$\mathcal{D}$$.

In practice, we optimize the MLE objective using mini-batch gradient ascent. The algorithm operates in iterations. At every iteration $$t$$, we sample a mini-batch $$\mathcal{B}_t$$ of datapoints sampled randomly from the dataset ($$\vert \mathcal{B}_t\vert < \vert \mathcal{D} \vert$$) and compute gradients of the objective evaluated for the mini-batch. These parameters at iteration $$t+1$$ are then given via the following update rule
{% math %}
\theta^{(t+1)} = \theta^{(t)} + r_t \nabla_\theta\mathcal{L}(\theta^{(t)} \vert \mathcal{B}_t)
{% endmath %}

where $$\theta^{(t+1)}$$ and $$\theta^{(t)}$$ are the parameters at iterations $$t+1$$ and $$t$$ respectively, and $$r_t$$ is the learning rate at iteration $$t$$. Typically, we only specify the initial learning rate $$r_1$$ and update the rate based on a schedule. [Variants](http://cs231n.github.io/optimization-1/) of stochastic gradient ascent, such as RMS prop and Adam, employ modified update rules that work slightly better in practice.

From a practical standpoint, we must think about how to choose hyperparameters (such as the initial learning rate) and a stopping criteria for the gradient descent. For both these questions, we follow the standard practice in machine learning of monitoring the objective on a validation dataset. Consequently, we choose the hyperparameters with the best performance on the validation dataset and stop updating the parameters when the validation log-likelihoods cease to improve[^1].

Now that we have a well-defined objective and optimization procedure, the only remaining task is to evaluate the objective in the context of an autoregressive generative model. To this end, we substitute the factorized joint distribution of an autoregressive model in the MLE objective to get

{% math %}
\max_{\theta \in \mathcal{M}}\frac{1}{\vert D \vert} \sum_{\mathbf{x} \in\mathcal{D} }\sum_{i=1}^n\log p_{\theta_i}(x_i \vert \mathbf{x}_{< i})
{% endmath %}

where $$\theta = \{\theta_1, \theta_2, \ldots, \theta_n\}$$ now denotes the
collective set of parameters for the conditionals.

Inference in an autoregressive model is straightforward. For density estimation of an arbitrary point $$\mathbf{x}$$, we simply evaluate the log-conditionals $$\log p_{\theta_i}(x_i \vert \mathbf{x}_{< i})$$ for each $$i$$ and add these up to obtain the log-likelihood assigned by the model to $$\mathbf{x}$$. Since we know conditioning vector $$\mathbf{x}$$, each of the conditionals can be evaluated in parallel. Hence, density estimation is efficient on modern hardware.

Sampling from an autoregressive model is a sequential procedure. Here, we first sample $$x_1$$, then we sample $$x_2$$ conditioned on the sampled $$x_1$$, followed by $$x_3$$ conditioned on both $$x_1$$ and $$x_2$$ and so on until we sample $$x_n$$ conditioned on the previously sampled $$\mathbf{x}_{< n}$$. For applications requiring real-time generation of high-dimensional data such as audio synthesis, the sequential sampling can be an expensive process. Later in this course, we will discuss how parallel WaveNet, an autoregressive model sidesteps this expensive sampling process.

<!-- TODO: add NADE samples figure -->

Finally, an autoregressive model does not directly learn unsupervised representations of the data. In the next few set of lectures, we will look at latent variable models (e.g., variational autoencoders) which explicitly learn latent representations of the data.

<!-- 

Additional parameterizations
==============
Coming soon: MADE, Char-RNN, Pixel-CNN, Wavenet -->

Footnotes
==============

[^1]: Given the non-convex nature of such problems, the optimization procedure can get stuck in local optima. Hence, early stopping will generally not be optimal but is a very practical strategy.
