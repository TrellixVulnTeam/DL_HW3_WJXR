r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""

### 1 ###

i)
The Jacobian tensor will be in shape of (in_features, N) * (out_features, N)
= (1024, 64) * (512, 64) = (1024, 64, 512, 64)

ii)
Yes, each sample is i.i.d from the 64 samples we have, so the derivative for the input 0,i to output 0,j = 0.
Because of that most of the elements in the Jacobian tensor will be zeros, 1024 * 512 * 64 * 63 = 2,113,929,216.

iii)
No. $\delta\mat{Y}*\pderiv{Y}{\mat{X}}$ = ($\delta\mat{X}$) is a known fact but we can
use the fact that $\pderiv{Y}{\mat{X}}$ = W which we have calculated

### 2 ###

i)
The Jacobian tensor will be in shape of W * X =
= (1024, 64) * (512, 64) = (1024, 64, 512, 1024)

ii)
Yes, for the same reason as the previous part. The derivative for the input 0,i to output 0,j = 0.

iii)
No. Just like the previous part, we will calculate ($\delta\mat{W}$) that
is the same as $\delta\mat{Y}*\pderiv{Y}{\mat{W}} = dot(\delta\mat{Y}, X)$
"""

part1_q2 = r"""
**Your answer:**

There are algorithms that don't require back-propagation or any  derivative at all.
For example: https://link.springer.com/article/10.1007%2Fs10898-012-9951-y

However, as noted in their abstract "We find that the ability of all these 
solvers to obtain good solutions diminishes with increasing problem size."

So, it seems that still GD with back-propagation is better for complex problems.

"""
# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.09, 0.07847, 0.011
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1099
    lr_vanilla = 0.0299
    lr_momentum = 0.02
    lr_rmsprop = 0.000833
    reg = 0.008

    # wstd = 0.094
    # lr_vanilla = 0.0220005
    # lr_momentum = 0.00300001
    # lr_rmsprop = 0.0001212111
    # reg = 0.01

    #best yet
    # wstd = 0.1099
    # lr_vanilla = 0.0299
    # lr_momentum = 0.02
    # lr_rmsprop = 0.000833
    # reg = 0.008
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr = 0.0025
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
Since we know that dropout is used to reduce overfitting, it is no surprise that when comparing 0 to higher dropout on 
the train set, the 0 dropout performs better on it (overfitting the training data). When comparing to the test results,
the 0.4 dropout achieved better results on the test. However, when using too high dropout (0.8), our test accuracy
dropped significantly. This is also no surprise since when we try to discard too much data with high dropout we hard our 
model's robustness (model hasn't seen enough quality samples to learn on, we discarded too much data).
"""

part2_q2 = r"""
**Yes, its possible** 
For the accuracy to change, its not enough for the scores to be less accurate. For example if the classify classes an 
object of class A to 100% probability or 80% probability, the classifier will still class the object as class A even
though the loss increases (the prediction is less "accurate") our accuracy but will change the loss. 
For a more concrete example, on the dropout graphs on epoch 28~ onwards, on the test set with dropout = 0, we can
see an increase to the loss and to the accuracy.
"""

part2_q3 = r"""

1) Back propagation and GD are two separate things.
GD is a an iterative optimization algorithm for finding the local minimum of a differentiable function.
While back-propagation computes the gradient of the loss function with respect to the weights of the
network for a single input–output example, and does so efficiently.

The main difference is that back-propagation refers only to the algorithm for computing the gradient
not how the gradient is used. GD refers to the algorithm that uses the gradient in a certain way.

2) In both gradient descent and stochastic gradient descent, you
update a set of parameters in an iterative manner to minimize an error function.

While in GD, you have to run through all the samples in your training set to do a 
single update for a parameter in a particular iteration, in SGD, on the other hand, you use a random subset 
of training sample from your training set to do the update for a parameter in a particular iteration.

3) SGD is more likely to get out of a local minimum since every random subset can have a slightly different
loss space. Also, SGD is faster since we use a smaller subset where in GD we use all of the training set.

4)

i) To calculate the backward-pass 

ii) For the backward stage we need to save in the memory the gradient results in the forward stage,
so even tough we split the data we need to save too much gradients simultaneously.  

"""
# ==============

# ==============
# Part 3 answers


def part3_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0.0,
        learn_rate=0.0,
        lr_sched_factor=0.0  ,
        lr_sched_patience=0,
    )
    # Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 256
    hypers['seq_len'] = 50
    hypers['h_dim'] = 1000
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.35
    hypers['lr_sched_patience'] = 2
    # ========================
    return hypers


def part3_generation_params():
    start_seq = ""
    temperature = 0.0005
    # Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    # ========================
    return start_seq, temperature


part3_q1 = r"""
We split the corpus into sequences for two reasons:
1. Solve the issue of vanishing gradients - If we dont split the corpus and suffer from vanishing gradients, the model
we will get as output will be useless.
2. Reduce RAM - If we dont split the corpus we can load the training samples gradually and our memory commitment is
reduced during the training process
"""

part3_q2 = r"""
The model saves and considers the last few inputs prior to the current (hidden states are propergated between batches),
and thus the model is able to understand and use the full contex and refer to prior sequences, and the text we will get
as output will be over the length of the original sequence length. 
"""

part3_q3 = r"""
Sentences have context in them, and it is important for understand the text to process the text in its original order.
If we shuffle the order of the batches, all the prior sentence/paragraph information will be lost and we will be training
on input that makes no logical sense, and we will not be able to generate any text of value.
"""

part3_q4 = r"""
1. Low temperatures incentivize the model to pick chars based on previous learning of the model High temperatures increase
the chance low score chars will be picked next.

2. As we stated in 1, if we have high temperature, our model will be incentivized to pick chars with lower scores more often,
   which will cause high prediction variance.
    
3. Very low temperature will cause the probability if low score chars to be chosen to be extremely low, meaning that our model will
   pick the higher score chars most of the time (if not all the time, causing identical char choice every iteration).
"""
# ==============
