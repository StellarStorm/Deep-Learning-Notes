+++
author = "Skylar"
title = "Gradient Descent Optimizers"
date = "2021-03-30"
description = "Optimizers."
tags = [
    "tensorflow",
    "pytorch",
    "code",
]
categories = [
    "deep learning",
]
+++

Optimizers are crucial to training a Deep Learning model, but (for me anyways)
they are neither chosen as carefully or even understood nearly so much as other
components such as loss. This is understandable - you can't visualize them the
same way you can a loss function, for instance - but it's a mistake. The
optimizer is what drives the learning process itself! The loss measures how
similar (or not) the output of your model is to the ground truth, but then it's
the job of the optimizer to adjust the model parameters so that the loss
can be decreased over the next epochs, over and over again until the loss
minima (and, hopefully, the optimal model) can be found. As such, it's
worthwhile to at least look at the basics of how commonly-used optimizers work
and get an intuition (if not perfect understanding) of the principles at play.

Optimizers work on a simple principle - finding the minima of the loss function.
Gradient descent moves towards that minima by calculating the gradient of the
loss function over the current parameters at a certain time point, such as the
end of a training epoch, then adjusting the parameters using the *negative* of
the gradient in order to go "downhill" towards that minima (remember, a gradient
points towards the steepest slope at a point, so to find the minima you'd want
to go in the opposite, or negative, direction).

Another way to think of this is to visualize the codomain of the loss function
over the existing model parameters as a multi-dimensional landscape of montains
and valleys, with "montains" corresponding to high loss (poor model performance)
and "valleys" corresponding to low loss (good model performance). We always want
the model to be making its way "down the montains" until it gets to the very
best it can, that is, until loss decreases as low as feasible. Thus, at any
particular step in training, these optimizers are calculating the gradient of
the loss function at the current model parameters. Then, since gradient points
"uphill", the optimizer just adjusts all the model parameters by the negative
of the gradient so that the model can decrease the loss in the next training
step.

BTW a **ton** of this information was sourced from the amazing blog posts by
[Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/index.html) and
[John Chen](https://johnchenresearch.github.io/demon/) -- huge thanks to them
for their great work!

A lot of papers use different notations, so to keep things consistent, I'm
adapting the following conventions:

- $\theta$ : the optimizer parameters
- $\eta$: the learning rate
- $J$: the loss function
- $g$: the gradient of the loss function, i.e. $g=\nabla J_\theta (\theta)$
- $m$: the average momentum of the past gradients (most sources that are dealing
    with non-adaptive optimizers, where there is only one momentum term, write
    this as $v$)
- $v$: the average momentum of the past gradients, squared (in implementation,
    this is multiplied by a decay constant and thus is **not** identical to
    $m^2$)
- $\beta$ or $\beta_1$: a decay rate for $m$ (other sources often write this
    term as $\mu$)
- $\beta_2$: a decay rate for for $v$

## SGD

Stochastic Gradient Descent is the "grandad" so to speak, the backbone that
all the other optimizers I'm looking at can trace their lineage from. Properly
speaking this is "mini-batch" SGD - that is, it updates on mini-batches of
training data rather than the entire batch all at once. This avoids expensive
and redundant calculations while providing the same information that a
large-scale gradient operation over the entire dataset would provide.

Essentially, SGD just calculates the gradient of the loss for a specific
mini-batch. Then, all the model parameters are adjusted by `learning rate *
gradient`.

#### Math

$$\theta_{t+1} = \theta_t - \eta g_t$$


#### Algorithm

> $\textbf{Initialize } \theta_0$ \
> $\textbf{While } \theta_t$ not converged \
> $\quad t \leftarrow t + 1$ \
> $\quad g_t \leftarrow \nabla_\theta J_t(\theta_{t - 1})$ \
> $\quad \textbf{Update }$\
> $\qquad \theta_t \leftarrow \theta_{t-1} - \eta g_t$


#### Links

[TensorFlow](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/optimizer_v2/gradient_descent.py#L27-L191),
[PyTorch](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD)

## SGD with momentum

On its own, SGD is good but often slow in terms of finding the minima of the loss
function for the dataset, struggling to find the best route to the minima and
sometimes confused by ravines in the loss landscape.

![SGD_no_momentum](/Deep-Learning-Notes/images/posts/grad_optim/wo_momentum.gif)
![SGD_momentum](/Deep-Learning-Notes/images/posts/grad_optim/with_momentum.gif)

*Left: SGD without momentum, right: SGD with momentum <cite>[1]</cite>*

One way to address this is adding in a "momentum" component that acts similarly
to momentum in real life - allowing the optimizer to pay less attention
to minor fluctuations and find the optimal route to the minima more quickly.

Intuitively, think of a styrofoam ball and a baseball, rolling down a hillside when they hit
a little dip. The styrofoam ball is light and has little momentum, so it might
get trapped by this dip and not roll all the way down. On the other hand,
the baseball has more momentum and will just roll over the upper edges of the
dip and keep on going downhill. This is very similar to what momentum does for
SGD (and other) optimizers -- it helps them avoid settling into local minima by,
in a sense, ignoring little fluctuations in the loss when a trend is pointing
elsewher. This is just like how the baseball saw a little "upwards" trend in the hillside
when it was inside the dip, but it was able to roll upwards for a small distance
to continue its downward path).

Specifically, "[t]he momentum term increases for dimensions whose gradients point
in the same directions and reduces updates for dimensions whose gradients
change directions. As a result, we gain faster convergence and reduced
oscillation." <cite>[1]</cite>

#### Math

1) $m_t = \beta m_{t-1} + \eta g_t$

2) $\theta_{t+1} = \theta_t - m_t$

By combining these expressions into a single step, you can see that this is
simply ordinary SGD minus a momentum term

$$\theta_{t+1} = \underbrace{(\theta_t - \eta g_t)}_{\text{SGD}} - \beta m_{t-1}$$

#### Algorithm

### Code Implementations

[TensorFlow](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/optimizer_v2/gradient_descent.py#L27-L191),
[PyTorch](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD)

## SGD with Nestorov momentum (NAG)

On its on, SGD + Momentum blindly follows the slope of the loss landscape.
We can try to improve this with Nesterov momentum, which gives an estimate of...

Intuitively, you can think of this as a four-step process <cite>[3]</cite>

1. Project the position of the solution
2. Calculate the gradient of the projection
3. Calculate the change in the variable using the partial derivative
4. Update the variable

#### Momentum vs NAG



![Nesterov](/Deep-Learning-Notes/images/posts/grad_optim/nesterov.jpeg)

*Figure courtesy Stanford CS213n <cite>[5]</cite>*

#### Math

1) $m_t = \beta m_{t-1} + \eta \nabla_\theta J(\theta - \beta m_{t-1})$
2) $\theta_{t+1} = \theta - m_t$

Or, combined into a single expression,

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta - \beta m_{t-1}) - \beta m_{t-1}$$

#### Algorithm

> $\textbf{Initialize } \theta_0, m_0 \leftarrow 0$ \
> $\textbf{While } \theta_t$ not converged \
> $\quad t \leftarrow t + 1$ \
> $\quad g_t \leftarrow \nabla_\theta J_t(\theta_{t - 1} - \eta \beta m_{t - 1})$ \
> $\quad m_t = \beta m_{t - 1} + g_t$\
> $\quad \textbf{Update }$\
> $\qquad \theta_t \leftarrow \theta_{t-1} - \eta m_t$

#### Algorithm v2

Dozat (2015) proposes a re-write of Nesterove momentum [4]. In this version a
warming step is assumed, so $\beta$ is parameterized by $t$ as well.

> $\textbf{Initialize } \theta_0, m_0 \leftarrow 0$ \
> $\textbf{While } \theta_t$ not converged \
> $\quad t \leftarrow t + 1$ \
> $\quad g_t \leftarrow \nabla_\theta J_t(\theta_{t - 1})$ \
> $\quad m_t = \beta_t m_{t - 1} + g_t$\
> $\quad \bar{m_t} = \beta_{t + 1} + m_t + g_t$\
> $\quad \textbf{Update }$\
> $\qquad \theta_t \leftarrow \theta_{t-1} - \eta \bar{m_t}$



#### Links

[On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.pdf)

[TensorFlow](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/optimizer_v2/gradient_descent.py#L27-L191),
[PyTorch](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD)

## Adaptive Optimizers

SGD optimizers are non-adaptive; that is, they use a global learning rate for
all parameters $\theta_t$ at timestep $t$. In contrast, adaptive optimizers
adjust the learning rate for each parameter $\theta_{i,t}$. This theoretically
allows for better performance or faster convergence. Despite this, practice
shows that SGD with (Nesterov) momentum is often just as good or better than
adaptive methods if given enough training time.

## Adam

Adam is one of, if not the most, popular adaptive optimizers (or just optimizers
in general). A simplistic view of Adam is that it is RMSProp (not discussed here) with
momentum.

#### Math

1) Set exponential decay rates $\beta_1$ and $\beta_2$ (often 0.9 and 0.999)
2) Calulate gradient $g_t$
3) Find the decaying average of past gradients, $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
4) Find the decaying average of past squared gradients, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
5) Correct $m_t$ so it is no longer biased towards zero, $\hat{m_t} = \frac{m_t}{1 - \beta_1^t}$
6) Correct $v_t$ so is it no longer biased towards zero, $\hat{v_t} = \frac{v_t}{1 - \beta_2^t}$
7) Use a small $\epsilon$ term to avoid division by zero

The update rule for Adam is

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}$$

Note that $\beta_1^t$ and $\beta_2^t$ denote raising the decay constants to the
power of $t$. These are only used in the bias correction steps.

#### Algorithm

> $\textbf{Initialize } \theta_0, m_0 \leftarrow 0, v_0 \leftarrow 0, t \leftarrow 0$ \
> $\textbf{While } \theta_t$ not converged \
> $\quad t \leftarrow t + 1$ \
> $\quad g_t \leftarrow \nabla_\theta J_t(\theta_{t - 1})$ \
> $\quad m_t \leftarrow \beta_1 m_{t-1} + (1-\beta_1) g_t$ \
> $\quad v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ \
> $\quad \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t}$ \
> $\quad \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t}$ \
> $\quad \textbf{Update }$\
> $\qquad \theta_t \leftarrow \theta_{t-1} - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}$

#### Links

[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

[TensorFlow](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/optimizer_v2/adam.py#L31-L250)

## AMSGrad

This modification of Adam is based on the belief that the exponential moving
averages of the gradients provide only short-term memory, which leads to Adam
being outperformed by SGD in some circumstances. Instead, the authors propose
keeping the largest update to the squared average gradient.

Note also one difference: there is no bias-correction for $m_t$ or $v_t$.

#### Math

1) Set exponential decay rates $\beta_1$ and $\beta_2$ (often 0.9 and 0.999)
2) Calulate gradient $g_t$
3) Find the decaying average of past gradients, $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
4) Find the decaying average of past squared gradients, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
5) Keep the largest update for $v_t$, $v_t$ so is it no longer biased towards zero, $\hat{v_t} = \max(\hat{v}_{t-1}, v_t)$
6) Use a small $\epsilon$ term to avoid division by zero

The update rule for AMSGrad is

$$\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{\hat{v_t}} + \epsilon}$$

#### Algorithm

> $\textbf{Initialize } \theta_0, m_0 \leftarrow 0, v_0 \leftarrow 0, t \leftarrow 0$ \
> $\textbf{While } \theta_t$ not converged \
> $\quad t \leftarrow t + 1$ \
> $\quad g_t \leftarrow \nabla_\theta J_t(\theta_{t - 1})$ \
> $\quad m_t \leftarrow \beta_1 m_{t-1} + (1-\beta_1) g_t$ \
> $\quad v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ \
> $\quad \hat{v_t} \leftarrow \max(\hat{v}_{t-1}, v_t)$ \
> $\quad \textbf{Update }$\
> $\qquad \theta_t \leftarrow \theta_{t-1} - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}$

#### Links

[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)

[Tensorflow](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/optimizer_v2/adam.py#L31-L250)

[PyTorch](https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam)

## AdaBelief

This modification to Adam attempts to adjust the size of the next
update based on the "belief" or confidence that that proposed update is correct.

Intuitively, you can think of $m_t$ as a prediction of $g_t$. When the
prediction is close (low variance) then AdaBelief takes a large step. However,
as the variance between prediction $m_t$ and observed $g_t$ becomes greater,
the optimizers performs smaller updates. That is, low variance means greater
belief in $g_t$ and a larger update; high variance means less belief and
a smaller update. In practice, I've observed that AdaBelief quickly outperformed
vanilla Adam in the early stages of model training for a (difficult)
segmentation tasks. However, after training for a sufficiently long period Adam
regained the lead and, at the inference stage, consistently yielded better
segmentation predictions then AdaBelief. It's entirely possible that this is
just a mis-configured example in my training, though, and AdaBelief might do
better with adjusted hyperparameters.

#### Algorithm

> $\textbf{Initialize } \theta_0, m_0 \leftarrow 0, v_0 \leftarrow 0, t \leftarrow 0$ \
> $\textbf{While } \theta_t$ not converged \
> $\quad t \leftarrow t + 1$ \
> $\quad g_t \leftarrow \nabla_\theta J(\theta_{t - 1})$ \
> $\quad m_t \leftarrow \beta_1 m_{t-1} + (1-\beta_1) g_t$ \
> $\quad v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) (g_t - m_t)^2 + \epsilon$ \
> $\quad \textbf{If } \textit{AMSGrad}$\
> $\qquad v_t \leftarrow max(v_t, v_{t-1})$ \
> $\quad \hat{m_t} \leftarrow \frac{m_t}{1 - \beta_1^t}$ \
> $\quad \hat{v_t} \leftarrow \frac{v_t}{1 - \beta_2^t}$ \
> $\quad \textbf{Update}$ \
> $\qquad \theta_t \leftarrow \theta_{t-1} - \frac{\eta \hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}$

### Math

#### Links
[AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468)

[TensorFLow (from original authors)](https://github.com/juntang-zhuang/Adabelief-Optimizer/blob/update_0.2.0/pypi_packages/adabelief_tf0.2.1/adabelief_tf/AdaBelief_tf.py)

[PyTorch (from original authors)](https://github.com/juntang-zhuang/Adabelief-Optimizer/blob/update_0.2.0/pypi_packages/adabelief_pytorch0.2.1/adabelief_pytorch/AdaBelief.py)

## Nadam

"Much like Adam is essentially RMSProp with momentum, Nadam is Adam with
Nesterov momentum."
 [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam)

#### Links

[Incorporating Nesterov Momentum into Adam](http://cs229.stanford.edu/proj2015/054_report.pdf)
[TensorFlow](https://github.com/tensorflow/tensorflow/blob/a4dfb8d1a71385bd6d122e4f27f86dcebb96712d/tensorflow/python/keras/optimizer_v2/nadam.py#L31-L219)


## References
[1] [Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/index.html)

[2] [John Chen](https://johnchenresearch.github.io/demon/)

[3] [Jason Brownlee](https://machinelearningmastery.com/gradient-descent-with-nesterov-momentum-from-scratch/)

[4] [Dozat (2015)](http://cs229.stanford.edu/proj2015/054_report.pdf)

[5] [Stanford CS231N](https://cs231n.github.io/neural-networks-3/#sgd)
