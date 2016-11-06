import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from adaline import AdalineGD


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(
        x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=cmap(idx), marker=markers[idx], label=cl)


# read the data
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values

# make the plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

#ada1 = AdalineGD(n_iter=10, eta=0.01).fit(x,y)
#ax[0].plot(range(1, len(ada1.cost_)+1),np.log10(ada1.cost_), marker = 'o')
#ax[0].set_xlabel('Epochs')
#ax[0].set_ylabel('log(Sum-squared-error)')
#ax[0].set_title('Adaline - Learning Rate 0.01')

#ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(x,y)
#ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
#ax[1].set_xlabel('Epochs')
#ax[1].set_ylabel('Sum-squared-error')
#ax[1].set_title('Adaline - Learning rate 0.0001')

# standardizing instead
x_std = np.copy(x)
x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(x_std, y)
plot_decision_regions(x_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
