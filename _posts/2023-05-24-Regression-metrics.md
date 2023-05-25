**Regression performance metrics** are used to evaluate how well a regression model fits the data. These metrics measure the accuracy and goodness-of-fit of the model and can help determine how well the model can predict new, unseen data. Some common regression performance metrics include:
1. Mean Absoulute Error (MAE)
2. Mean Squared Error (MSE)
3. Root Mean Squared Error (RMSE)
4. R-squared (R^2) or Coefficient of Determination
5. Adjusted R-squared:

To learn about these, let's go with a Simple Linear Regression example with a simple dataset.
X is an independent variable whereas y is a dependent variable
View the dataframe's top 5 rows. You can skip the Python code, if and wherever you want. 

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
Below are the top 5 rows out of 40

![alt text](/assets/images/postsImages/2.RegressionMetrics/xandy.png){: style="width:100%; float:centre;"}

We will see how the dataframe looks like when drawn on a chart through a scatterplot, the python code of which is below

 ```tsql
 sns.scatterplot(data = data, x = 'X', y = 'y', color = 'darkblue')
plt.xlabel('X', backgroundcolor = 'skyblue')
plt.ylabel('y', rotation = 'horizontal', backgroundcolor = 'skyblue')
plt.title('Randomly distributed datapoints', backgroundcolor = 'skyblue')
plt.show()
```
![alt text](/assets/images/postsImages/2.RegressionMetrics/randompoints2.png){: style="width:100%; float:centre;"}


Now we normally fit a best fit line on all the 40 points
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
That's how a best fit line looks like

![alt text](/assets/images/postsImages/2.RegressionMetrics/randompoints1.png){: style="width:100%; float:centre;"}


For even better understanding, let's only take 10 datapoints, I know it's not much, but just for the simplicity and clear visualisation, consider that our dataset has 10 datapoints and let's draw a line of best fit on those datapoints

```tsql

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
Below is what the 10 points with a best-fit line look like. The dotted lines are the distance of all the respective points from the line. The dotted line is parallel to the y-axis.

![alt text](/assets/images/postsImages/2.RegressionMetrics/randompoints.png){: style="width:100%; float:centre;"}

Below is just the Python code which creates a table that calculates the difference between y and y-predictions

```tsql
data['y_pred'] = y_pred
data['y - y_pred'] = abs(y - y_pred)

data.head(10).style.set_properties(**{'text-align': 'center', 'color': 'black', 'width': '120px', 'font-size': '11pt'})\
.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', 'skyblue'), ('color', 'black'), ('font-weight', 'bold')]},
                   {'selector': '.row_heading, .blank', 'props': [('color', 'black'), ('font-weight', 'bold')]}])\
.apply(lambda x: ['background-color: orange' if i == x[-1] else 'background-color: skyblue' for i in x], axis=1)

```

This is how the new table looks like

![alt text](/assets/images/postsImages/2.RegressionMetrics/yminusypred.png){: style="width:100%; float:centre;"}

Below is the Python code for calculating the mean of y-y_pred column.

```tsql
data['y - y_pred'].mean() 
```

Output: 0.47666945948143774


Which is nothing but **MAE**

### 1. Mean Absolute Error (MAE):

The MAE measures the average absolute difference between the predicted values and the actual values. It is also a good choice for continuous numerical data and is less sensitive to outliers than MSE. The formula for MAE is:

$$ 
𝑀𝐴𝐸=(1/𝑛)∗∑|𝑦𝑖−ŷ𝑖|
$$

#### Advantages Of MAE:

1. MAE is easy to understand and interpret since it represents the average magnitude of errors.
2. Unlike the mean squared error (MSE), MAE does not heavily penalize large errors and is more robust to outliers.
3. MAE is a good choice when the target variable has a linear relationship with the predictors.

#### Disadvantages:

1. Since MAE does not heavily penalize large errors, it may not be the best metric to use when the goal is to minimize the maximum error (i.e., minimize the worst-case scenario).
2. In some cases, the absolute difference may not be an appropriate measure of the error. For example, if the predicted and actual values are both negative but with different magnitudes, the MAE may not accurately represent the error.
3. MAE does not provide any information about the direction of the error (i.e., overprediction or underprediction).



Now, working on our table further we calculate the respective squares of the y-y_pred column and then take a mean of this entire column

```tsql
data['(y - y_pred)^2'] = data['y - y_pred'] ** 2

data.head(10).style.set_properties(**{'text-align': 'center', 'color': 'black', 'width': '120px', 'font-size': '11pt'})\
.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center'), ('background-color', 'skyblue'), ('color', 'black'), ('font-weight', 'bold')]},
                   {'selector': '.row_heading, .blank', 'props': [('color', 'black'), ('font-weight', 'bold')]}])\
.apply(lambda x: ['background-color: orange' if i == x[-1] else 'background-color: skyblue' for i in x], axis=1)

```

![alt text](/assets/images/postsImages/2.RegressionMetrics/yminusypredsq.png){: style="width:100%; float:centre;"}

```tsql
data['(y - y_pred)^2'].mean()
```

Output: 0.2765202084983117

Which is nothing but **MSE**

### 2. Mean Squared Error (MSE):

The MSE measures the average of the squared differences between the predicted values and the actual values. It penalizes larger errors more heavily than smaller ones, making it a good choice for continuous numerical data. The formula for MSE is:

$$
𝑀𝑆𝐸=(1/𝑛)∗∑(𝑦𝑖−ŷ𝑖)²
$$

where n is the number of samples, yi is the actual value, and ŷi is the predicted value.

or you can use mean_squared_error function from sklearn library to calculate directly.

```tsql
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)
print(mse)
```
Output: 0.2765202084983117

which is of course the same.

#### Advantages:

1. It penalizes larger errors more heavily than smaller errors due to the squaring operation.
2. It is a differentiable and convex function, which makes it easier to optimize using numerical methods.
3. It is widely used in many applications and is a standard metric for regression models.
4. It provides a measure of variance in addition to the measure of bias provided by the squared bias.

#### Disadvantages:

1. It is highly sensitive to outliers because of the squaring operation, which can result in a large increase in the error for a single outlier.
2. It is not in the same units as the original target variable, which makes it difficult to interpret the error in a meaningful way.
3. It can be heavily influenced by large errors, which may not be representative of the overall performance of the model.
4. It tends to prioritize accuracy over interpretability, which may not be desirable in some applications.


### 3. Root Mean Squared Error (RMSE):
The RMSE is the square root of the MSE and is also a good choice for continuous numerical data. The formula for RMSE is:

<!-- $$
𝑅𝑀𝑆𝐸=𝑠𝑞𝑟𝑡((1/𝑛)∗∑(𝑦𝑖−ŷ𝑖)²)
$$

<br>
<br>
 -->

$$
𝑅𝑀𝑆𝐸 = \sqrt{1/n{\sum{(𝑦𝑖−ŷ𝑖)²}}}
$$

```tsql
rmse = mean_squared_error(y, y_pred, squared = False)
print(rmse)
```
Output: 0.5258518883662125


#### Advantages:

1. RMSE is widely used and is easily interpretable, making it useful for communicating model performance to non-technical stakeholders.
2. It provides a measure of the magnitude of errors in the same units as the response variable, which makes it easier to understand how large the errors are.
3. It punishes larger errors more than smaller errors, which can be useful in situations where larger errors are more important to avoid.

#### Disadvantages:

1. RMSE is sensitive to outliers, meaning that a few large errors can significantly increase the value of RMSE and make it difficult to interpret.
2. Because RMSE involves taking the square root of the average squared error, it is more difficult to compute than simpler metrics like MAE.
3. RMSE can be heavily influenced by the scale of the response variable, meaning that comparisons across different response variables may not be meaningful.

### 4. R-squared (R²):
The R-squared value measures the proportion of variance in the dependent variable (y) that can be explained by the independent variables (X) in the model. It ranges from 0 to 1, with a higher value indicating a better fit. The formula for R-squared is:

$$
𝑅²=1−(𝑆𝑆𝑟𝑒𝑠/𝑆𝑆𝑡𝑜𝑡)
$$

where SSres is the sum of squared residuals

$$
SS_{\text{res}} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$


& SStot is the total sum of squares.

$$
SS_{\text{tot}} = \sum_{i=1}^{n}(y_i - \bar{y})^2
$$

it can also be written as

$$
R^2 = 1 - \left(\frac{\sum_{i=1}^{n}(y_i - \hat{y}i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}\right)
$$

Or you could import r2_score from sklearn.metrics in Python and calculate directly

```tsql
from sklearn.metrics import r2_score
R2 = r2_score(y, y_pred)
print(R2)
```
Output: 0.28998880211940314


#### Advantages:

1. It provides a simple and straightforward way to evaluate the goodness of fit of a regression model.
2. It ranges from 0 to 1, where a value of 1 indicates a perfect fit and a value of 0 indicates that the model does not explain any of the variance in the dependent variable.
3. It is widely used in practice and easily interpretable.

#### Disadvantages:

1. It can be misleading when used with models that have low degrees of freedom or when the relationship between the independent and dependent variables is non-linear.
2. It does not indicate whether the independent variables are causally related to the dependent variable or whether they are merely correlated.
3. It can be sensitive to outliers, especially when the sample size is small. In such cases, the adjusted R-squared metric is often preferred.


### 5. Adjusted R-squared:
The Adjusted R-squared value adjusts the R-squared value to account for the number of independent variables in the model. It penalizes models with more independent variables and can help prevent overfitting. The formula for Adjusted R-squared is:

$$
𝐴𝑑𝑗𝑢𝑠𝑡𝑒𝑑𝑅²=1−[(𝑛−1)/(𝑛−𝑝−1)]∗(1−𝑅²)
$$

$$
Adjusted R^2 = 1 - \left(\frac{n-1}{n-p-1}\right) \cdot (1-R^2)
$$


where n is the number of samples and p is the number of independent variables.

Below is the Python code to calculate Adjusted Rsquared via r2_score 

```tsql
from sklearn.metrics import r2_score
R2 = r2_score(actual, predicted)
Adj_r2 = 1-(1-R2)*(n-1)/(n-p-1)
```

#### Advantages:

1. It is a useful tool for comparing models with different numbers of independent variables, as it adjusts for the number of independent variables used in the model.
2. It provides a more accurate representation of the goodness of fit of the model than the R-squared value alone.

#### Disadvantages:

1. It can be negative, which means that the model is worse than the baseline model (i.e., the model that predicts the mean of the dependent variable).
2. It assumes that the independent variables are linearly related to the dependent variable and that there are no interactions between the independent variables. If these assumptions are violated, the adjusted R-squared may not be an accurate measure of model fit.
<br>
<br>
<br>
