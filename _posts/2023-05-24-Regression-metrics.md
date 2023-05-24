Regression performance metrics are used to evaluate how well a regression model fits the data. These metrics measure the accuracy and goodness-of-fit of the model and can help determine how well the model can predict new, unseen data. Some common regression performance metrics include:

To learn what are the performence metrics of Linear Regression are, let's go with a Simple Linear Regression Example with a simple dataset.
X is an independent variable whereas y is a dependent variable
View the dataframe's top 5 rows

```tsql
np.random.seed(0)
X = np.random.rand(40, 1)
y = 2 * X.squeeze() + 0.5 * np.random.randn(40)

data = pd.DataFrame({'X': X.squeeze(), 'y': y})
# data.head()

data.head(5).style.set_properties(**{'text-align': 'center', 'color': 'black', 'width': '120px', 'font-size': '11pt'})\
.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', 'skyblue'), ('color', 'black'), ('font-weight', 'bold')]},
                   {'selector': '.row_heading, .blank', 'props': [('color', 'black'), ('font-weight', 'bold')]}])\
.applymap(lambda x: f'background-color: skyblue' if x else '')

```

![alt text](/assets/images/postsImages/2.RegressionMetrics/xandy.png){: style="width:70%; float:centre;"}

We will see how the dataframe looks like when drawn on a chart through a scatterplot

 ```tsql
 sns.scatterplot(data = data, x = 'X', y = 'y', color = 'darkblue')
plt.xlabel('X', backgroundcolor = 'skyblue')
plt.ylabel('y', rotation = 'horizontal', backgroundcolor = 'skyblue')
plt.title('Randomly distributed datapoints', backgroundcolor = 'skyblue')
plt.show()
```
![alt text](/assets/images/postsImages/2.RegressionMetrics/randompoints1.png){: style="width:90%; float:centre;"}

```tsql
np.random.seed(0)
X = np.random.rand(40, 1)
y = 2 * X.squeeze() + 0.5 * np.random.randn(40)

data = pd.DataFrame({'X': X.squeeze(), 'y': y})

sns.scatterplot(data=data, x='X', y='y', color='darkblue')
plt.xlabel('X', backgroundcolor='skyblue')
plt.ylabel('y', rotation='horizontal', backgroundcolor='skyblue')
plt.title('Randomly distributed datapoints', backgroundcolor='skyblue')

# Fit a linear regression model
reg = LinearRegression().fit(X, y)
# Generate predictions
y_pred = reg.predict(X)
# Plot the best fit line
plt.plot(X, y_pred, color='red', linewidth=2)

plt.show()
```

![alt text](/assets/images/postsImages/2.RegressionMetrics/randompoints1.png){: style="width:90%; float:centre;"}


For better understanding, let's only take 10 datapoints, I know it's not much, but just for the simplicity and clear visualisation, consider that our dataset has 10 datapoints and let's draw a line of best fit on those datapoints

```tsql
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# np.random.seed(0)
X = np.random.rand(10, 1)
y = 2 * X.squeeze() + 0.5 * np.random.randn(10)

data = pd.DataFrame({'X': X.squeeze(), 'y': y})

sns.scatterplot(data=data, x='X', y='y', color='darkblue')
plt.xlabel('X', backgroundcolor='skyblue')
plt.ylabel('y', rotation='horizontal', backgroundcolor='skyblue')
plt.title('Randomly distributed datapoints', backgroundcolor='skyblue')

# Fit a linear regression model
reg = LinearRegression().fit(X, y)
# Generate predictions
y_pred = reg.predict(X)

plt.plot(X, y_pred, color='red', linewidth=2)

for i in range(len(X)):
    x_val = X[i][0]
    y_val = y[i]
    y_pred_val = y_pred[i]
    plt.plot([x_val, x_val], [y_val, y_pred_val], color='grey', linestyle='--')
    # Plot the coordinates of every data point
    plt.annotate(f"({x_val:.2f}, {y_val:.2f})", xy=(x_val, y_val), xytext=(x_val+0.02, y_val-0.1), fontsize=8)

plt.show()

```

![alt text](/assets/images/postsImages/2.RegressionMetrics/randompoints1.png){: style="width:90%; float:centre;"}

```tsql
data['y_pred'] = y_pred
data['y - y_pred'] = abs(y - y_pred)

data.head(10).style.set_properties(**{'text-align': 'center', 'color': 'black', 'width': '120px', 'font-size': '11pt'})\
.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', 'skyblue'), ('color', 'black'), ('font-weight', 'bold')]},
                   {'selector': '.row_heading, .blank', 'props': [('color', 'black'), ('font-weight', 'bold')]}])\
.apply(lambda x: ['background-color: orange' if i == x[-1] else 'background-color: skyblue' for i in x], axis=1)

```

![alt text](/assets/images/postsImages/2.RegressionMetrics/yminusypred.png){: style="width:90%; float:centre;"}


data['y - y_pred'].mean() 

Output: 0.47666945948143774

Which is nothing Buy MAE

Mean Absolute Error (MAE):
The MAE measures the average absolute difference between the predicted values and the actual values. It is also a good choice for continuous numerical data and is less sensitive to outliers than MSE. The formula for MAE is:

$$ 
𝑀𝐴𝐸=(1/𝑛)∗∑|𝑦𝑖−ŷ𝑖|
$$

Advantages Of MAE:

MAE is easy to understand and interpret since it represents the average magnitude of errors.
Unlike the mean squared error (MSE), MAE does not heavily penalize large errors and is more robust to outliers.
MAE is a good choice when the target variable has a linear relationship with the predictors.
Disadvantages:

Since MAE does not heavily penalize large errors, it may not be the best metric to use when the goal is to minimize the maximum error (i.e., minimize the worst-case scenario).
In some cases, the absolute difference may not be an appropriate measure of the error. For example, if the predicted and actual values are both negative but with different magnitudes, the MAE may not accurately represent the error.
MAE does not provide any information about the direction of the error (i.e., overprediction or underprediction).


Mean Squared Error

```tsql
data['(y - y_pred)^2'] = data['y - y_pred'] ** 2

data.head(10).style.set_properties(**{'text-align': 'center', 'color': 'black', 'width': '120px', 'font-size': '11pt'})\
.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', 'skyblue'), ('color', 'black'), ('font-weight', 'bold')]},
                   {'selector': '.row_heading, .blank', 'props': [('color', 'black'), ('font-weight', 'bold')]}])\
.apply(lambda x: ['background-color: orange' if i == x[-1] else 'background-color: skyblue' for i in x], axis=1)

```

![alt text](/assets/images/postsImages/2.RegressionMetrics/yminusypredsq.png){: style="width:90%; float:centre;"}

```tsql
data['(y - y_pred)^2'].mean()
```

Output: 0.2765202084983117

Which is nothing but MSE

Mean Squared Error (MSE):
The MSE measures the average of the squared differences between the predicted values and the actual values. It penalizes larger errors more heavily than smaller ones, making it a good choice for continuous numerical data. The formula for MSE is:

$$
𝑀𝑆𝐸=(1/𝑛)∗∑(𝑦𝑖−ŷ𝑖)²
$$

where n is the number of samples, yi is the actual value, and ŷi is the predicted value.

mean_squared_error can be used to calculate directly

```tsql
mse = mean_squared_error(y, y_pred)
print(mse)
```
Output: 0.2765202084983117

Below are some Advantages and Disadvantages of Mean Squared Error

Advantages:

It penalizes larger errors more heavily than smaller errors due to the squaring operation.
It is a differentiable and convex function, which makes it easier to optimize using numerical methods.
It is widely used in many applications and is a standard metric for regression models.
It provides a measure of variance in addition to the measure of bias provided by the squared bias.
Disadvantages:

It is highly sensitive to outliers because of the squaring operation, which can result in a large increase in the error for a single outlier.
It is not in the same units as the original target variable, which makes it difficult to interpret the error in a meaningful way.
It can be heavily influenced by large errors, which may not be representative of the overall performance of the model.
It tends to prioritize accuracy over interpretability, which may not be desirable in some applications.

Root Mean Squared Error (RMSE):
The RMSE is the square root of the MSE and is also a good choice for continuous numerical data. The formula for RMSE is:

𝑅𝑀𝑆𝐸=𝑠𝑞𝑟𝑡((1/𝑛)∗∑(𝑦𝑖−ŷ𝑖)²)

```tsql
rmse = mean_squared_error(y, y_pred, squared = False)
print(rmse)
```
Output: 0.5258518883662125

Below are some Advantages and Disadvantages of Root Mean Squared Error

Advantages:

RMSE is widely used and is easily interpretable, making it useful for communicating model performance to non-technical stakeholders.
It provides a measure of the magnitude of errors in the same units as the response variable, which makes it easier to understand how large the errors are.
It punishes larger errors more than smaller errors, which can be useful in situations where larger errors are more important to avoid.
Disadvantages:

RMSE is sensitive to outliers, meaning that a few large errors can significantly increase the value of RMSE and make it difficult to interpret.
Because RMSE involves taking the square root of the average squared error, it is more difficult to compute than simpler metrics like MAE.
RMSE can be heavily influenced by the scale of the response variable, meaning that comparisons across different response variables may not be meaningful.

R-squared (R²):
The R-squared value measures the proportion of variance in the dependent variable (y) that can be explained by the independent variables (X) in the model. It ranges from 0 to 1, with a higher value indicating a better fit. The formula for R-squared is:

𝑅²=1−(𝑆𝑆𝑟𝑒𝑠/𝑆𝑆𝑡𝑜𝑡)
where SSres is the sum of squared residuals

SSres = ((y - y_pred)**2).sum()

& SStot is the total sum of squares.

SStot = ((y - y.mean())**2).sum()

```tsql
R2 = r2_score(y, y_pred)
print(R2)
```
Output: 0.28998880211940314


Below are some Advantages and Disadvantages of R-squared:

Advantages:

It provides a simple and straightforward way to evaluate the goodness of fit of a regression model.
It ranges from 0 to 1, where a value of 1 indicates a perfect fit and a value of 0 indicates that the model does not explain any of the variance in the dependent variable.
It is widely used in practice and easily interpretable.
Disadvantages:

It can be misleading when used with models that have low degrees of freedom or when the relationship between the independent and dependent variables is non-linear.
It does not indicate whether the independent variables are causally related to the dependent variable or whether they are merely correlated.
It can be sensitive to outliers, especially when the sample size is small. In such cases, the adjusted R-squared metric is often preferred.


Adjusted R-squared:
The Adjusted R-squared value adjusts the R-squared value to account for the number of independent variables in the model. It penalizes models with more independent variables and can help prevent overfitting. The formula for Adjusted R-squared is:

𝐴𝑑𝑗𝑢𝑠𝑡𝑒𝑑𝑅²=1−[(𝑛−1)/(𝑛−𝑝−1)]∗(1−𝑅²)
where n is the number of samples and p is the number of independent variables.


```tsql
# Adj_r2 = 1-(1-R2)*(n-1)/(n-p-1)
# print(Adj_r2)
```

Advantages of adjusted R-squared:

It is a useful tool for comparing models with different numbers of independent variables, as it adjusts for the number of independent variables used in the model.
It provides a more accurate representation of the goodness of fit of the model than the R-squared value alone.
Disadvantages of adjusted R-squared:

It can be negative, which means that the model is worse than the baseline model (i.e., the model that predicts the mean of the dependent variable).
It assumes that the independent variables are linearly related to the dependent variable and that there are no interactions between the independent variables. If these assumptions are violated, the adjusted R-squared may not be an accurate measure of model fit.

Mean Squared Log Error (MSLE):
It measures the average of the squared logarithmic differences between predicted and actual values. It is useful when the target variable has a wide range of values. The formula for MSLE is:

𝑀𝑆𝐿𝐸=1/𝑛∗∑(𝑙𝑜𝑔(𝑦+1)−𝑙𝑜𝑔(ŷ+1))²

```tsql
# MSLE = mean_squared_log_error(y, y_pred)
# print(MSLE)
```

Advantages:

It is a good metric to use when the target variable has a large range of values, as it scales the differences between predictions and actual values based on the log of the target variable.
It punishes large differences between the predicted and actual values more heavily than smaller differences, which can be useful in certain applications.
It is less sensitive to outliers than mean squared error (MSE), which can make it a better metric to use when the dataset contains outliers.
Disadvantages:

It is not interpretable in the same way as other metrics such as mean absolute error (MAE) or R-squared, which can make it difficult to explain to non-technical stakeholders.
It can be difficult to compare MSLE scores across different datasets, as the metric depends on the scale of the target variable.
It can be sensitive to zero values in the actual values, as the logarithm of zero is undefined. This can be addressed by adding a small constant to the actual values before calculating the metric.


Mean absolute percentage error (MAPE)
The mean absolute percentage error (MAPE) is a commonly used evaluation metric in forecasting and time series analysis. It measures the average percentage deviation of the predicted values from the actual values.

𝑀𝐴𝑃𝐸=(1/𝑛)∗Σ(|(𝑦𝑖−ŷ𝑖)/𝑦𝑖|)∗100
where:

n: number of observations

```tsql
MAPE = mean_absolute_percentage_error(y, y_pred)
print(MAPE)
```
Output: 2.82017066869959

Advantages:

It is easy to understand and interpret, as it provides a percentage error.
It is scale-independent, which means it can be used to compare the accuracy of models that are predicting values on different scales.
It is widely used in forecasting and time series analysis literature, and it is therefore easy to find references for comparisons.
Disadvantages:

It has an undefined value when the actual value is zero, which can happen frequently in some applications.
It gives a higher weight to larger errors, which can be problematic when the actual values have small magnitudes.
It does not work well when there are extreme values or outliers in the data, as they can distort the percentage error.
It can lead to misleading interpretations when the actual values are close to zero or the model predicts zero values

Mean Percentage Error (MPE):
It measures the average percentage difference between predicted and actual values. The formula for MPE is:

𝑀𝑃𝐸=1/𝑛∗∑(𝑦−ŷ)/𝑦

Advantages of Mean Percentage Error (MPE):

Provides a percentage measure of the forecast error, which is more interpretable than absolute error measures.
Can be used to compare the accuracy of different forecasting methods.
Disadvantages of Mean Percentage Error (MPE):

Can produce biased results if the time series has zero or negative values, since the denominator in the calculation of the percentage error would be zero or negative.
Not as popular as other error measures like MAE, MSE, RMSE, and MAPE.
Does not take into account the magnitude of the errors, so it can be misleading if the errors have a wide range of values.

