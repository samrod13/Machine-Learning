import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron

# Read in the iris dataset
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# df.tail()
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values
#plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
#plt.scatter(x[50:100, 0], x[50:100, 1], color='blue',
#            marker='x', label='versicolor')
#plt.xlabel('petal length')
#plt.ylabel('sepal length')
#plt.legend(loc='upper left')
#plt.show()

# Training perceptron algorithm on the Iris dataset
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)

# plot out the results
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
