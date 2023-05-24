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

![alt text](/assets/images/postsImages/2.RegressionMetrics/xandy.png){: style="width:200px; float:centre;"}

We will see how the dataframe looks like when drawn on a chart through a scatterplot

 ```tsql
 sns.scatterplot(data = data, x = 'X', y = 'y', color = 'darkblue')
plt.xlabel('X', backgroundcolor = 'skyblue')
plt.ylabel('y', rotation = 'horizontal', backgroundcolor = 'skyblue')
plt.title('Randomly distributed datapoints', backgroundcolor = 'skyblue')
plt.show()
```
![alt text](/assets/images/postsImages/2.RegressionMetrics/randompoints1.png){: style="width:80%; float:centre;"}

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





