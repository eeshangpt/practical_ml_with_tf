# Loss Function in ML

## How to estimate parameters of a model

Lets us again consider 
1. *LINEAR REGRESSION* model.  
    $$ h_{w,b}(x) = y = b + w_1 x_1 + w_2 x_2 + ... + w_m x_m $$ 
    compactly written as $$ h_{w,b}(x) = y = b + \Sigma_{i=1}^m w_i x_i $$

2. *LOGISTIC REGRESSION* model.
    $$Pr(y = 1 |x) = \frac{1}{1 + e^{-z}}$$ 
    where $$ z = b + \Sigma_{i=1}^m w_i x_i $$
Now that we have data and labels, our job is to come with values of paramenters i.e. $b$ and $w$'s

Let us assume a single variable LINEAR REGRESSION model with bias $b = 0$.
$$ \Rightarrow y = b + w_1 x_1 $$ or $$ y = w_1 x_1 $$
We have a lot of data points, and for different value of $w_1$ we get different lines passing through origin.  
**Loss function** is denoted by $J(w, b)$.  
Loss function in case of LINEAR REGRESSION is calculated as distance of each point from the model or the line. Mathematically, for a single point, error of $i^{th}$ point = $(h_{w,b}(x^{(i)}) - y^{(i)})^2$.  
So, $$ J(w,b) = \frac{1}{2} \Sigma_{i=1}^n(h_{w,b}(x^{(i)}) - y^{(i)})^2$$  
>Here $\frac{1}{2}$ is multiplied only for mathematical convenience.

In case of Linear Regression. $$J(w,b) = \frac{1}{2} \Sigma_{i=1}^n([b + w_1 x_1^{(i)}] - y^{(i)})^2$$

**CASE**: LOGISTIC REGRESSION 

Considering binary classification scenario.

| ACTUAL ($y$) | PREDICTED ($\hat{y}$) | Comment |
| --- | --- | --- |
| 1 | 0 | Error |
| 0 | 1 | Error |
| 1 | 1 | OK (no error) |
| 0 | 0 | OK (no error) |


Loss when $y = 1$ is $-y \log(p)$ and when $y = 0$ loss is $-(1 - y) \log(1 - p)$ where $\hat{y} \Leftrightarrow p$. 

We get Cross-entropy loss $$ = -y \log(p) - (1 -y) \log(1 - p)$$

>For Multi-class classification poblem we use *Categorical Cross-Entropy Loss* and *Sparse Categorical Cross-Entropy Loss*

To find the value of parameter, we need to minimize loss and find **optimal** values of parameters. This is where **optimization techniques** are used. 
