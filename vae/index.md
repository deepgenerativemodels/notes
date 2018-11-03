---
layout: post
title: Variational Autoencoders
---

{% math %}
\newcommand{\D}{\mathcal{D}}
\newcommand{\KL}[2]{D_\mathrm{KL}\paren{#1 \mathbin{\|} #2}}
\newcommand{\P}{\mathcal{P}}
\newcommand{\X}{\mathcal{X}}
\newcommand{\Z}{\mathcal{Z}}
\newcommand{\Q}{\mathcal{Q}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\M}{\mathcal{M}}
\newcommand{\ELBO}{\mathrm{ELBO}}
\newcommand{\bz}{\mathbf{z}}
\newcommand{\giv}{\mid}
\newcommand{\paren}[1]{\left(#1\right)}
\newcommand{\brac}[1]{\left[#1\right]}
\newcommand{\veps}{\varepsilon}
\newcommand{\set}[1]{\left\{#1\right\}}
\renewcommand{\d}{\mathop{}\!\mathrm{d}}
\newcommand{\Expect}{\mathbb{E}}
\newcommand{\Normal}{\mathcal{N}}
\newcommand{\I}{\mathbf{I}}
\newcommand{\0}{\mathbf{0}}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
{% endmath %}

The goal of this post is to provide a look at some of the core mathematical principles of the variational autoencoder. We shall do so by developing the variational autoencoder "from scratch", starting with a latent variable model and culminating with the use of deep neural networks to train the generative model with a variational procedure.

We begin by considering a directed latent variable model of the following form
{% math %}
p(\bx, \bz) = p(\bx \giv \bz)p(\bz),
{% endmath %}
where $$\bz$$ is the latent variable and $$\bx$$ is the observed variable. This model describes a generative process that samples $$\bx$$ using the procedure
{% math %}
\begin{align}
\bz &\sim p(\bz) \\
\bx &\sim p(\bx \giv \bz).
\end{align}
{% endmath %}
As such, we shall call this our generative model. If one adopts the belief that the latent variable $$\bz$$ somehow encodes semantically meaningful information about $$\bx$$, it is natural to view this generative process as first generating the "high-level" semantic information about $$\bx$$ first before fully generating $$\bx$$. Such a perspective motivates generative models with rich latent variable structures such as hierarchical generative models $$p(\bx, \bz_1, \ldots, \bz_m) = p(\bx \giv \bz_1)\prod_i p(\bz_i \giv \bz_{i+1})$$---where information about $$\bx$$ is generated hierarchically---and temporal models such as the Hidden Markov Model---where temporally-related high-level information is generated first before constructing $$\bx$$.

We now consider a family of distributions $$\P_\bz$$ where $$p(\bz) \in \P_\bz$$ describes a probability distribution over $$\bz$$. Next, consider a family of conditional distributions $$\P_{\bx\giv \bz}$$ where $$p(\bx \giv \bz) \in \P_{\bx\giv \bz}$$ describes a conditional probability distribution over $$\bx$$ given $$\bz$$. Then our hypothesis class of generative models is the set of all possible combinations
{% math %}
\begin{align}
\P_{\bx,\bz} = \set{p(\bx, \bz) \giv p(\bz) \in \P_\bz, p(\bx \giv \bz) \in \P_{\bx\giv\bz}}.
\end{align}
{% endmath %}
Given a dataset $$\D = \set{\bx^{(1)}, \ldots, \bx^{(n)}}$$, we are interested in the following learning and inference tasks
- Selecting $$p \in \P_{\bx,\bz}$$ that "best" fits $$\D$$.
- Approximate inference of $$\bz$$: given a sample $$\bx$$, how do we impute its latent variable $$\bz$$?
<!-- - Approximate marginal inference of $$\bx$$: given partial access to certain dimensions of the vector $$\bx$$, how do we impute the missing parts? -->

We shall also assume the following
- Intractability: computing the posterior probability $$p(\bz \giv \bx)$$ is intractable.
- Big data: the dataset $$\D$$ is too large to fit in memory; we can only work with small, sub-sampled batches of $$\D$$.

Learning Directed Latent Variable Models
==============

One way to measure how closely $$p(\bx, \bz)$$ fits the data $$\D$$ is to measure the Kullback-Leibler (KL) divergence between the data distribution (which we denote as $$p_\D(\bx)$$) and the model's marginal distribution $$p(\bx) = \int p(\bx, \bz) \d \bz$$,
{% math %}
\begin{align}
\KL{p_\D(\bx)}{p(\bx)}.
\end{align}
{% endmath %}
The distribution that ``best'' fits is thus $$p \in \P_{\bx, \bz}$$ that minimizes the KL divergence. Equivalently, we wish to solve the maximum marginal log-likelihood problem
{% math %}
\begin{align}
\max_{p \in \P_{\bx, \bz}} \sum_{\bx \in \D} \log \int p(\bx, \bz) \d \bz.
\end{align}
{% endmath %}
However, this problem is intractable due to the difficulty of marginalizing the latent variable---which is as difficult as computing the posterior $$p(\bz \mid \bx)$$. Rather than maximizing the log-likelihood directly, we instead construct a lower bound that is more amenable to optimization. To do so, we introduce a variational family $$\Q$$ and note that the following relationships hold true[^1] for all proposal distributions $$q(\bz) \in \Q$$
{% math %}
\begin{align}
\log \int p(\bx, \bz) \d \bz &= \log \int \frac{q(\bz)}{q(\bz)} p(\bx, \bz) \d \bz\\
&\ge\int q(\bz) \log \frac{p(\bx, \bz)}{q(\bz)} \d \bz,
\end{align}
{% endmath %}
where the inequality arises from Jensen's inequality. The gap between the LHS (marginal log-likelihood) and the RHS (Evidence Lower Bound) is captured by the KL divergence $$\KL{q(\bz)}{p(\bz \giv \bx)}$$. The inequality is tight when the proposed distribution $$q(\bz)$$ exactly matches $$p(\bz \giv \bx)$$. In contrast to the marginal log-likelihood, the ELBO admits a tractable unbiased estimator
{% math %}
\begin{align}
\sum_i q(\bz^{(i)}) \log \frac{p(\bx, \bz)}{q(\bz^{(i)})} \text{, where } \bz^{(i)} \sim q(\bz),
\end{align}
{% endmath %}
so long as it is easy to sample from and compute densities for $$q(\bz)$$. Since the ELBO itself presents an optimization problem, the maximum marginal log-likelihood problem is now replaced with
{% math %}
\begin{align}
\max_{p \in \P_{\bx, \bz}} \sum_{\bx \in \D} \paren{\max_{q \in \Q} \Expect_{q(\bz)} \log \frac{p(\bx, \bz)}{q(\bz)}}.
\end{align}
{% endmath %}


Black-Box Variational Inference
==============

In this post, we shall focus on first-order stochastic gradient methods for optimizing the ELBO. These optimization techniques are desirable in that they allow us to sub-sample the dataset during optimization---but require our objective function to be differentiable with respect to the optimization variables. As such, we shall posit for now that any $$p(\bx, \bz) \in \P_{\bx, \bz}$$ and $$q(\bz) \in \Q$$ are alternatively parameterizable as $$p_\theta(\bx, \bz)$$ and $$q_\lambda(\bz)$$ and that these distributions are differentiable with respect to $$\theta$$ and $$\lambda$$.

This inspires an Expectation-Maximization-like algorithm, where, for each mini-batch $$\M = \set{\bx^{(1)}, \ldots, \bx^{(m)}}$$, the following two steps are performed.

**Step 1**

We first do *per-sample* optimization of $$q$$ by iteratively applying the update
{% math %}
\begin{align}
\lambda^{(i)} \gets \lambda^{(i)} + \tilde{\nabla}_\lambda \ELBO(\bx^{(i)}; \theta, \lambda^{(i)}),
\end{align}
{% endmath %}
where $$\text{ELBO}(\bx; \theta, \lambda) = \Expect_{q_\lambda(\bz)} \log \frac{p_\theta(\bx, \bz)}{q_\lambda(\bz)}$$, and $$\tilde{\nabla}_\lambda$$ denotes an unbiased estimate of the ELBO gradient. This step seeks to approximate the log-likelihood $$\log p_\theta(\bx^{(i)})$$.

**Step 2**

We then perform a single update step based on the mini-batch
{% math %}
\begin{align}
\theta \gets \theta + \tilde{\nabla}_\theta \sum_{i} \ELBO(\bx^{(i)}; \theta, \lambda^{(i)}),
\end{align}
{% endmath %}
which corresponds to the step that hopefully moves $$p_\theta$$ closer to $$p_\D$$.

A Note on Gradient Estimation
==============

The gradients $$\nabla_\lambda \ELBO$$ and $$\nabla_\theta \ELBO$$ can be estimated via Monte Carlo sampling. While it is straightforward to construct an unbiased estimate of $$\nabla_\theta \ELBO$$ by simply pushing $$\nabla_\theta$$ through the expectation operator, the same cannot be said for $$\nabla_\lambda$$. Instead, we see that
{% math %}
\begin{align}
\nabla_\lambda \Expect_{q_\lambda(\bz)} \log \frac{p_\theta(\bx, \bz)}{q_\lambda(\bz)} = \Expect_{q_\lambda(\bz)} \brac{\paren{\log \frac{p_\theta(\bx, \bz)}{q_\lambda(\bz)}} \cdot \nabla_\lambda \log q_\lambda(\bz)}.
\end{align}
{% endmath %}
This equality follows from the log-derivative trick (also commonly referred to as the REINFORCE trick). The full derivation involves some simple algebraic manipulations and is left as an exercise for the reader. The gradient estimator $$\tilde{\nabla}_\lambda \ELBO$$ is thus
{% math %}
\begin{align}
\frac{1}{k}\sum_{i=1}^k \brac{\paren{\log \frac{p_\theta(\bx, \bz^{(i)})}{q_\lambda(\bz^{(i)})}} \cdot \nabla_\lambda \log q_\lambda(\bz^{(i)})} \text{, where } \bz^{(i)} \sim q_\lambda(\bz).
\end{align}
{% endmath %}
However, it is often noted that this estimator suffers from high variance. One of the key contributions of the variational autoencoder paper is the reparameterization trick, which introduces an auxiliary distribution $$p(\veps)$$ and a differentiable function $$T(\veps; \lambda)$$ such that the procedure
{% math %}
\begin{align}
\veps &\sim p(\veps)\\
\bz &\gets T(\veps; \lambda),
\end{align}
{% endmath %}
is equivalent to sampling from $$q_\lambda(\bz)$$. By the Law of the Unconscious Statistician, we can see that
{% math %}
\begin{align}
\nabla_\lambda \Expect_{q_\lambda(\bz)} \log \frac{p_\theta(\bx, \bz)}{q_\lambda(\bz)} = \Expect_{p(\veps)} \nabla_\lambda \log \frac{p_\theta(\bx, T(\veps; \lambda))}{q_\lambda(T(\veps; \lambda))}.
\end{align}
{% endmath %}
In contrast to the REINFORCE trick, the reparameterization trick is often noted empirically to have lower variance and thus results in more stable training. 
<!-- \rs{I think there exists pathological examples where REINFORCE has lower variance than reparamterization. Should we talk about that?} -->

Deep Generative Model Parameterization
============== 

So far, we have described $$p_\theta(\bx, \bz)$$ and $$q_\lambda(\bz)$$ in the abstract. To instantiate these objects, we consider choices of parametric distributions for $$p_\theta(\bz)$$, $$p_\theta(\bx \giv \bz)$$, and $$q_\lambda(\bz)$$. A popular choice for $$p_\theta(\bz)$$ is the unit Gaussian
{% math %}
\begin{align}
p_\theta(\bz) = \Normal(\bz \giv \0, \I).
\end{align}
{% endmath %}
An popular alternative is a mixture of Gaussians with trainable mean and covariance parameters. 

The conditional distribution $$p_\theta(\bx \giv \bz)$$ is where we introduce a deep neural network. We note that a conditional distribution can be constructed by defining a distribution family (parameterized by $$\omega \in \Omega$$) in the target space $$\bx$$ (i.e. $$p_\omega(\bx)$$ defines an unconditional distribution over $$\bx$$) and a mapping function $$g_\theta: \Z \to \Omega$$. It is natural to call $$g_\theta$$ the decoder that is parameterized by $$\theta$$. The act of conditioning on $$\bz$$ is thus equivalent to using the choice of $$\omega = g(\bz)$$.In other words, $$g_\theta$$ defines the conditional distribution
{% math %}
\begin{align}
    p_\theta(\bx \giv \bz) = p_\omega(\bx) \text{ , where } \omega = g_\theta(\bz).
\end{align}
{% endmath %}
The generative model $$p_\theta(\bx, \bz)$$ is called a *deep* generative model since we will be using a neural network to instantiate the function $$g_\theta$$. In the case where $$p_\theta(\bx \giv \bz)$$ is a Gaussian distribution, we can thus represent it as
{% math %}
\begin{align}
    p_\theta(\bx \giv \bz) = \Normal(\bx \giv \mu_\theta(\bz), \Sigma_\theta(\bz)),
\end{align}
{% endmath %}
where $$\mu_\theta(\bz)$$ and $$\Sigma_\theta(\bz)$$ are neural networks that propose the mean and covariance matrix for the Gaussian distribution over $$\bx$$ when conditioned on $$\bz$$.

Finally, the variational family for the proposal distribution $$q_\lambda(\bz)$$ needs to be chosen judiciously so that the reparameterization trick is possible. Once again, a popular choice is the Gaussian distribution, where
{% math %}
\begin{align}
    \lambda &= (\mu, \Sigma) \\
    q_\lambda(\bz) &= \Normal(\bz \giv \mu, \Sigma)\\
    p(\veps) &= \Normal(\bz \giv \0, \I) \\
    T(\veps; \lambda) &= \mu + \Sigma^{1/2}\veps,
\end{align}
{% endmath %}
where $$\Sigma^{1/2}$$ is the Cholesky decomposition of $$\Sigma$$. For simplicity, practitioners often restrict $$\Sigma$$ to be a diagonal matrix (which restricts the distribution family to that of factorized Gaussians).

Amortized Variational Inference
==============

A noticable limitation of black-box variational inference is that **Step 1** executes an optimization subroutine that is computationally expensive. Recall that the goal of the **Step 1** is to find
{% math %}
\begin{align}
    \lambda^* = \argmax_{\lambda} \ELBO(\bx; \theta, \lambda).
\end{align}
{% endmath %}
For a given choice of $$\theta$$, there is a well-defined mapping from $$\bx \mapsto \lambda^*$$. A key realization is that this mapping can be *learned*. In particular, one can train an encoding function (parameterized by $$\phi$$) $$f_\phi: \X \to \Lambda$$ (where $$\Lambda$$ is the space of $$\lambda$$ parameters) on the following objective
{% math %}
\begin{align}
    \max_{\phi} \sum_{\bx \in \D} \ELBO(\bx; \theta, f_\phi(\bx)).
\end{align}
{% endmath %}
It is worth noting at this point that $$f_\phi(\bx)$$ can be interpreted as defining the conditional distribution $$q_\phi(\bz \giv \bx)$$. With a slight abuse of notation, we define
{% math %}
\begin{align}
    \ELBO(\bx; \theta, \phi) = \Expect_{q_\phi(\bz \mid \bx)} \log \frac{p_\theta(\bx, \bz)}{q_\phi(\bz \giv \bx)}.
\end{align}
{% endmath %}
and rewrite the optimization problem as 
{% math %}
\begin{align}
    \max_{\phi} \sum_{\bx \in \D} \ELBO(\bx; \theta, \phi).
\end{align}
{% endmath %}
It is also worth noting that optimizing $$\phi$$ over the entire dataset as a *subroutine* everytime we sample a new mini-batch is clearly not reasonable. However, if we believe that $$f_\phi$$ is capable of quickly adapting to always give a close-enough approximation of $$\lambda^*$$ given the current choice of $$\theta$$, then we can interleave the optimization $$\phi$$ and $$\theta$$. The yields the following procedure, where for each mini-batch $$\M = \set{\bx^{(1)}, \ldots, \bx^{(m)}}$$, we perform the following two updates jointly
{% math %}
\begin{align}
    \phi &\gets \phi + \tilde{\nabla}_\phi \sum_{\bx \in \M} \ELBO(\bx; \theta, \phi) \\
    \theta &\gets \theta + \tilde{\nabla}_\theta \sum_{\bx \in \M} \ELBO(\bx; \theta, \phi),
\end{align}
{% endmath %}
rather than running BBVI's **Step 1** as a subroutine. By leveraging the learnability of $$\bx \mapsto \lambda^*$$, this optimization procedure amortizes the cost of variational inference. If one further chooses to define $$f_\phi$$ as a neural network, the result is the variational autoencoder.


Footnotes
==============

[^1]: The first equality only holds if the support of $$q$$ includes that of $$p$$. If not, it is an inequality.
