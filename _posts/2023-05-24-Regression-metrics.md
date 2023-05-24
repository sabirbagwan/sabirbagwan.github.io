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

![alt text](//assets/images/postsImages/2.RegressionMetrics/xandy.png){: style="width:200px; float:right;"}
