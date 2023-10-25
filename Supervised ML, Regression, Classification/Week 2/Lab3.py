import matplotlib.pyplot as plt
import numpy as np

from lab_utils_multi import load_house_data

np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price 1000's")
plt.show()
