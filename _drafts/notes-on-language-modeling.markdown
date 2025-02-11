---
layout: post
title:  "Notes on Language Modeling"
date:   2025-01-28
categories: theory
---


# Language Modeling

In general, all types of language modeling train models to learn the next-token conditional distribution $\mathbb{P}(X|C)$, where $X=\{ x_1,\cdots,x_m\}$ is the set of tokens and $C$ is the set of contexts.
Where language models differ is how they condition on the context $C$.

If we have the distribution $\mathbb{P}(X|C)$, we can predict the probability of a sequence, and do decoding.


Left to right Language Modeling
---
This is a very common formulation of language-modeling where we train a model to predict the next token conditioned on the previously observed tokens.

$$
\mathbb{P}(x_1,\cdots,x_t) = \prod_{i<t}\mathbb{P}(x_i|x_{<i})
$$


Bidirectional Language Modeling
---

TODO


Masked Language Modeling
---

TODO

