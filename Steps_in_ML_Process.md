# Steps in Machine Learning Process.

## ML Process
### 1. Data pre-processing:
- Data comes from different sources and hence may be a little erroneous.
- Inconsistencies and outliers are removed to get high quality data. 
- Using high quality data for training is key to get a successful ML algo.
- Common steps in data pre-processing:
    - Normalize the features and bring them in same scale:
        - Z-Score: For a feature $X_j$ we calculate mean $\mu_j$ and standard deviation $\sigma_j$
            $$X_{j_{new}} = \frac{X_j - \mu_j}{\sigma_j}$$
            New feature ranges roughly $\in (-3, 3)$
        - Find $min_j$ and $max_j$, then,
            $$X_{j_{new}} = \frac{X_j - min_j}{max_j - min_j}$$
            New feature ranges roughly $\in (0, 1)$
        - Besides above stated normalization technques, *Log transformation*, and *Square-root transforation* are also used.
^
1. Visualize the data and explore it. Helps in crucial understanding of relation between feature and labels.

### 2. Model building.
1. Simplest model in **Linear Regression**.
    $$ y = b + w_1 x_1 + w_2 x_2 + ... + w_m x_m $$
   where there are m-features, and $(b, w_1, w_2, ..., w_m)$ are parameters.
   - Mapping between features and the label.
   - Geometrically the model represents a hyperplane.
2. **Logistic Regression**: Prediction of discrete quantities. 
    $$z = b + w_1 x_1 + w_2 x_2 + ... + w_m x_m$$
    - $z$ will be a real number and we want a discrete quantity $\in (0, 1)$, so we use logistic function.
    - One of a logistic function is *sigmoid function*
        $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
    - $\sigma(z)$ lies $\in (0, 1)$, which can easily be interpreted as probablity.
    - Therefore, we can write $$ Pr(y^{(i)} = 1 | x^{(i)}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$ where $$z = b + w_1 x_1 + w_2 x_2 + ... + w_m x_m$$
    - Fine in case of linearly separable features. If classes are non-linearly separable, we move to polynomial features, which is cumbersome. 
3. **Neural Network**: Feed-forward Neural Network.
    - Mechanism of constructing complex functions by considering simpler functions.
    ```
            h11
    x1          h21
            h12   h22     
            h13   h23     o1
    x2
            h14
    
    i/p     hidden       o/p
    ```
    - We begin with 2 features but number of paraments increase with each connection. 
    ```
    b  -----------+        +-------------+------------+
                    |        | Linear      | Non-linear |
    x1 -----------+------->| Combination | Activation |------> output
                    |        |             |            |
    x2 -----------+        +-------------+------------+
    
    ```
    - Here $b$ is bias, $x_1$ and $x_2$ are features. Linear Combination is simply $$ z = b + w_1 x_1 + w_2 x_2 $$ and Non-linear activation is any non-linear function most commonly **Rectified Linear Unit (ReLU)** for hidden layers and **Sigmoid Function** for output unit.
        $$ ReLU(z) = 
        \begin{array}{cc}
          \{ & 
            \begin{array}{cc}
              z & z > 0 \\
              0 & otherwise \\
            \end{array}
        \end{array}
        $$
    - Non-linear actiations are used to find non-linear boundaries.
