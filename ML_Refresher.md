# Machine Learning Refresher.

## ML from programming perspective.

**Problem 1**: Program to add 2 numbers ("a" and "b")  
    ```
    func f(a, b)
        return a + b
    ```

**Problem 2**: Recognizing hand written digits  
Can a prg be written as we have written as we have for the above problem.  
We can start by thinking of rules that can be stated for each digit, but are they scalable?? For eg., what if the orientation of the digit changes? Then the rules cannot cater to all the situation.

#### Analysis of problems and their solutions. 
It is difficult to write step wise solution to P2. In case of P1 we know the formula and the output is based on formula only.  
We can recognize the digits using our vision, because we are learning to read these digit as soon as we start our formal education. Also we have seen these digits written by many different people, so we have knowledge of different styles and orientations.  

*Question* that the ML tries to explore is that can mimic the training that we provided to our brain, to train a computer ?

**Traditional Programming World**
```
            +---------+
Data ------>|         |
            |         |
            | Program |-------> Output
            |         |
Rules ----->|         |
            +---------+
```

**Typical ML Operation**  
In this case we images of handwritten digits with their labels (i.e. number of the digit)
```
            +----------+
Data ------>|          |
            |          |
            |    ML    |-------> Rules / Patterns / Models
            |          |
Output ---->|          |
            +----------+
```
> **Key Difference**: In traditional programming paradigm the rules are part of the input, but in ML paradigm the rules are the output.  

**Final ML Procedure**
```
Data             Labels / Outputs
----             ----------------
 |                      |
 |    +------------+    |
 +--->| ML Trainer |<---+
      +------------+
            |
            |
            |
        +-------+    
New --->| Model |-----> Output
Data    +-------+
```

The process is 2 staged:
1. Training Phase: Where "model" is learned.
2. Inference / Prediction Phase: Similar to traditional programming paradigm.

## Terminology
1. Data:
    > Data is the new oil.
    - Data has to parts:
        - Feature
        - Label
    - Examples: Handwriiten images. $i^{th}$ image represented as $X^{(i)}$. *Image* is the feature and *digits 0-9* are the label  
    $D = \{ X^{(1)}, X^{(2)}, ... , X^{(n)}\}$ 
    - Example: House price predition. Features are house feature, label is price.
        - Feature of a house can be number of bedrooms, area in sq. ft., distance from school etc. 
        - $j^{th}$ feature for the $i^{th}$ data point is represented as $X^{(i)}_{j}$
    - So any data point is represented as $X^{(i)} : X^{(i)}_1, X^{(i)}_2, ..., X^{(i)}_m$ and label for this data point is represented as $y^{(i)}$
    - Data $D = \{(X^{(i)}, y^{(i)})\}_{i=1}^n$
2. Label: Based on label the type of algorithm/technique is defined.
    - If the label is absent, then the algorithm is Unsupervised Learning.
        - Eg: Cluster students based on attribute and many more 
    - If the label is present, then the algorithm is Supervised Learning. 
        - Eg: Handwitten digit recognization, Housing price prediction, and many more.
        - **Discrete Labels**: Classification problem. 
        - **Continuous Labels**: Regression Problem.
3. Feautes: Different type of features:
    - Numeric: eg. # of bedroom, area etc.
    - Categorical: eg. city names, colour names. These have to be converted to numbers, using *One-hot Encoding*.
    - **One-Hot Encoding** Example:  
    `City = {"Mumbai", "Delhi", "Chennai"}`
    | Feat | F_Mumbai | F_Delhi | F_Chennai |
    | --- | --- | --- | --- |
    | Mumbai | 1 | 0 | 0 |
    | Delhi | 0 | 1 | 0 |
    | Chennai | 0 | 0 | 1 |
