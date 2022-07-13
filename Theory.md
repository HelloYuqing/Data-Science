# Theoretical Questions 

Lengend:  😊Easy   😒Medium  😵‍💫Expert

# Table of contents

* [Supervised machine learning](https://github.com/HelloYuqing/Data-Science/blob/main/technical.md#supervised-machinelearning)
* [Linear regression](
* Validation
* Clssification
* Regularization
* Feature selection
* Decision trees
* Random forest
* Gradient boosting
* Parameter turning
* Netural networks
* Optimization in neural networks
* Neural networks for computer vision
* Text classification
* Clustering
* Dimensionality reduction
* Ranking and search
* Recommender systems
* Time series




## Supervised machine learning

**😊What is supervised machine learning?**

Supervised learning is a type of machine learning in which our algorithms are trained using well-labeled training data, and machines predict the output based on that data. Labeled data indicates that the input data has already been tagged with the appropriate output. Basically, it is the task of learning a function that maps the input set and returns an output. Some of its examples are: Linear Regression, Logistic Regression, KNN, etc.


## Linear regression

**😊what is regression? Which models can you use to solve a regression problems?**

Regression is a part of supervised ML. Regression models investigate the relationship between a dependent (target) and independent variable (s).

- *Linear Regression* establishes a linear relationship between target and predictor (s). It predicts a numeric value and has a shape of a straight line.
- *Polynomial Regression* has a regression equation with the power of independent variable more than 1. It is a curve that fits into the data points.
- *Ridge Regression* helps when predictors are highly correlated (multicollinearity problem). It penalizes the squares of regression coefficients but doesn’t allow the coefficients to reach zeros (uses L2 regularization).
- *Lasso Regression* penalizes the absolute values of regression coefficients and allows some of the coefficients to reach absolute zero (thereby allowing feature selection).


**😊What is linear regression? When do we use it?**

Linear regression is a model that assumes a linear relationship between the input variables (X) and the single output variable (y).

Simple linear regression
```
y = B0 + B1*x1 
```
Multiple linear regression
```
y = B0 + B1*x1 + ... + Bn * xN
```


**😊What are the main assumptions of linear regression?**

There are several assumptions of linear regression. If any of them is violated, model predictions and interpretation may be worthless or misleading.

1. **Linear relationship** between features and target variable.
2. **Additivity** means that the effect of changes in one of the features on the target variable does not depend on values of other features. For example, a model for predicting revenue of a company have of two features - the number of items _a_ sold and the number of items _b_ sold. When company sells more items _a_ the revenue increases and this is independent of the number of items _b_ sold. But, if customers who buy _a_ stop buying _b_, the additivity assumption is violated.
3. Features are not correlated (no **collinearity**) since it can be difficult to separate out the individual effects of collinear features on the target variable.
4. Errors are independently and identically normally distributed (y<sub>i</sub> = B0 + B1*x1<sub>i</sub> + ... + error<sub>i</sub>):
   1. No correlation between errors (consecutive errors in the case of time series data).
   2. Constant variance of errors - **homoscedasticity**. For example, in case of time series, seasonal patterns can increase errors in seasons with higher activity.
   3. Errors are normaly distributed, otherwise some features will have more influence on the target variable than to others. If the error distribution is significantly non-normal, confidence intervals may be too wide or too narrow.



**😒What’s the normal distribution? Why do we care about it?**

The normal distribution is a continuous probability distribution whose probability density function takes the following formula:

![formula](https://mathworld.wolfram.com/images/equations/NormalDistribution/NumberedEquation1.gif)

where μ is the mean and σ is the standard deviation of the distribution.

The normal distribution derives its importance from the **Central Limit Theorem**, which states that if we draw a large enough number of samples, their mean will follow a normal distribution regardless of the initial distribution of the sample, i.e **the distribution of the mean of the samples is normal**. It is important that each sample is independent from the other.

This is powerful because it helps us study processes whose population distribution is unknown to us.



**😒How do we check if a variable follows the normal distribution?**

1. Plot a histogram out of the sampled data. If you can fit the bell-shaped "normal" curve to the histogram, then the hypothesis that the underlying random variable follows the normal distribution can not be rejected.
2. Check Skewness and Kurtosis of the sampled data. Skewness = 0 and kurtosis = 3 are typical for a normal distribution, so the farther away they are from these values, the more non-normal the distribution.
3. Use Kolmogorov-Smirnov or/and Shapiro-Wilk tests for normality. They take into account both Skewness and Kurtosis simultaneously.
4. Check for Quantile-Quantile plot. It is a scatterplot created by plotting two sets of quantiles against one another. Normal Q-Q plot place the data points in a roughly straight line.



**😒What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices?**

Data is not normal. Specially, real-world datasets or uncleaned datasets always have certain skewness. Same goes for the price prediction. Price of houses or any other thing under consideration depends on a number of factors. So, there's a great chance of presence of some skewed values i.e outliers if we talk in data science terms. 

Yes, you may need to do pre-processing. Most probably, you will need to remove the outliers to make your distribution near-to-normal.


**😒What methods for solving linear regression do you know?**

To solve linear regression, you need to find the coefficients <img src="https://render.githubusercontent.com/render/math?math=\beta"> which minimize the sum of squared errors.

Matrix Algebra method: Let's say you have `X`, a matrix of features, and `y`, a vector with the values you want to predict. After going through the matrix algebra and minimization problem, you get this solution: <img src="https://render.githubusercontent.com/render/math?math=\beta = (X^{T}X)^{-1}X^{T}y">. 

But solving this requires you to find an inverse, which can be time-consuming, if not impossible. Luckily, there are methods like Singular Value Decomposition (SVD) or QR Decomposition that can reliably calculate this part <img src="https://render.githubusercontent.com/render/math?math=(X^{T}X)^{-1}X^{T}"> (called the pseudo-inverse) without actually needing to find an inverse. The popular python ML library `sklearn` uses SVD to solve least squares.

Alternative method: Gradient Descent. See explanation below.



**😒What is gradient descent? How does it work?**

Gradient descent is an algorithm that uses calculus concept of gradient to try and reach local or global minima. It works by taking the negative of the gradient in a point of a given function, and updating that point repeatedly using the calculated negative gradient, until the algorithm reaches a local or global minimum, which will cause future iterations of the algorithm to return values that are equal or too close to the current point. It is widely used in machine learning applications.



**😊What is the normal equation?**

Normal equations are equations obtained by setting equal to zero the partial derivatives of the sum of squared errors (least squares); normal equations allow one to estimate the parameters of a multiple linear regression.



**😊What is SGD  —  stochastic gradient descent? What’s the difference with the usual gradient descent?**

In both gradient descent (GD) and stochastic gradient descent (SGD), you update a set of parameters in an iterative manner to minimize an error function.

While in GD, you have to run through ALL the samples in your training set to do a single update for a parameter in a particular iteration, in SGD, on the other hand, you use ONLY ONE or SUBSET of training sample from your training set to do the update for a parameter in a particular iteration. If you use SUBSET, it is called Minibatch Stochastic gradient Descent.


**😊Which metrics for evaluating regression models do you know?**

1. Mean Squared Error(MSE)
2. Root Mean Squared Error(RMSE)
3. Mean Absolute Error(MAE)
4. R² or Coefficient of Determination
5. Adjusted R²



**😊What are MSE and RMSE?**

MSE stands for <strong>M</strong>ean <strong>S</strong>quare <strong>E</strong>rror while RMSE stands for <strong>R</strong>oot <strong>M</strong>ean <strong>S</strong>quare <strong>E</strong>rror. They are metrics with which we can evaluate models.



**😊What is the bias-variance trade-off?**

**Bias** is the error introduced by approximating the true underlying function, which can be quite complex, by a simpler model. **Variance** is a model sensitivity to changes in the training dataset.

**Bias-variance trade-off** is a relationship between the expected test error and the variance and the bias - both contribute to the level of the test error and ideally should be as small as possible:

```
ExpectedTestError = Variance + Bias² + IrreducibleError
```

But as a model complexity increases, the bias decreases and the variance increases which leads to *overfitting*. And vice versa, model simplification helps to decrease the variance but it increases the bias which leads to *underfitting*.


































































































