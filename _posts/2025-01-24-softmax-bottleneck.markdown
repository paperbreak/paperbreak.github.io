---
layout: post
title:  "Breaking the Softmax Bottleneck"
date:   2025-01-24
categories: theory
---

[Breaking the Softmax Bottleneck:
A High-Rank RNN Language Model](https://arxiv.org/abs/1711.03953)

# Background

When doing Language Modeling, we try and train a model to learn the conditional next-token distribution $\Pr(X \vert C)$, where $X= \lbrace x_1, \dots , x_m \rbrace$ is our vocabulary (set of tokens) and $C = \lbrace c_1, \dots, c_n \rbrace$ is the set of contexts. What the set $C$ looks like is based on the type of language modeling we want to perform, for example left-to-right or masked language modeling. In general, the context is just the tokens surrounding a particular token.

We can visualize this distribution as a matrix, with the number of rows being equal to the number of unique tokens $m$, and the number of columns being equal to the number of unique contexts $n$. Each entry at position $(i, j)$ represents the probability of token $x_i$ appearing in context $c_j$.

The way we train our NN to fit this distribution is in the following way:

$$
\begin{align}
\log \Pr(X \vert C) & \approx & \log \Pr(X \vert C ; \theta) \\
& = & \log \text{Softmax} ( H_\theta W_\theta^\top ) \\
\end{align}
$$

Where $W_\theta$ is our embedding weights, and $H_\theta$ is our context vectors (output of the final layer of our NN *before* the output embedding).
Essentially, this boils down to a matrix factorization problem - we are trying to approximate the theoretical distribution on the left as a product of two
matricies on the right.

Looking deeper, when we initialize the embedding weights $W_\theta$, we have to also pick their dimension size $d$. 
Therefore $W_\theta \in \mathbb{R}^{m \times d}$ and $H_\theta \in \mathbb{R}^{n \times d}$. This means we are performing a 
[Rank Factorization](https://en.wikipedia.org/wiki/Rank_factorization) of $\Pr(X \vert C)$, and $\text{rank}(\Pr(X \vert C)) = d$.

But this is now a bit concerning - in practice, we usually set $d$ to be a relatively low number, for example 786 for BERT.
This begins to imply the existance of the **Softmax Bottleneck**: if we set our embedding dimension $d$ too low, we will not
be able to accurately approximate the true distribution $\Pr(X \vert C)$.

# Formal Justification

## Definitions

1) Let $A = \log \Pr^*(X \vert C)$

Note: here we use the notation $\Pr^*$ to denote the true distribution. In practice, we don't know what the true distribution is - we are trying to solve for it.

2) Let $F(A)$ be the set of *row-wise* shifts of the matrix $A$:

$$
\begin{align}
F(A) & = \{ A + C \space \vert \space C_{i,:} = c, c \in \mathbb{R}  \} \\
& = \{ A + \Lambda J_{m,n} \space \vert \space \Lambda \text{ is diagonal}, \Lambda \in \mathbb{R}^{n \times n} \}
\end{align}
$$

At first glance, the set $F(A)$ will seem very unnatural. # TODO -- finish this

Note: I wrote two equivalent definitions for $A$ for (3) and (4) - definition (3) is easier to understand,
and definition (4) is more useful for one of our proofs later.

## Properties

1) For any matrix $A'$, $A' \in F(A) \iff \text{Softmax}(A') = \Pr^*(X \vert C)$

The set $F(A)$ defines the set of all logits that correspond to the true probability distribution $ \Pr^* $.  This means our logits $H_\theta W_\theta^\top$ have to be in the set $F(A)$ for our NN to be able to approximate the true distribution $ \Pr^* $.


$\implies$

$$
\begin{aligned}
    &  A' &\in& F(A) \\
    \implies & A' &=& A + C \\
    \implies & \text{Softmax}(A') &=& \text{Softmax}(A + C) \\
    & &=& \text{Softmax}(A) \\
    & &=& \text{Softmax}(\log \Pr^*(X \vert C)) \\
    & &=& \Pr^*(X \vert C) \quad \blacksquare
\end{aligned}
$$

Note: Softmax is shift-invariant (line 3)


$\impliedby$
$$
\begin{aligned}
& \text{Softmax}(A') &=& \Pr^*(X \vert C) \\
\implies & \log \text{Softmax}(A') &=& \log \Pr^*(X \vert C) \\
\implies & \log \text{Softmax}(A') &=& A \\
\implies & A' + \log \Sigma \exp(A') &=& A \\
\implies & A' &=& A - \log \Sigma \exp(A') \\
& &\in& \space F(A) \quad \blacksquare
\end{aligned}
$$


2) $\forall A^1 \neq A^2 \in F(A), \vert \text{rank}(A^1) - \text{rank}(A^2) \vert \leq 1$

This is saying that all matrices that belong to the set $F(A)$ have a similar rank. This is not surprising - every matrix in $F(A)$ is simply *row-wise* shifted by some constant factor.

$\implies$

TODO -- finish