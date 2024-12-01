# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

# import dataset
iris = datasets.load_iris()
X= iris.data
y = iris.target

# we are using small dataset, randomly choose 30 points and them them 
indices = np.random.choice(len(X), 30)
X=X[indices]
y=y[indices]
#print(y)

# plotting it in 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1,figsize=(20,15))
ax = Axes3D(fig,elev=48,azim=134)
ax.scatter(X[:,0], X[:, 1], X[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s = X[:, 3]*50)

for name, label in [('Virginica', 0), ('Setosa', 1),('Versicolour', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean(),
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w',facecolor='w'),size=25)

# set plot title and labels
ax.set_title("3D visualization", fontsize=40)
ax.set_xlabel("Sepal Length [cm]", fontsize=25)
ax.set_ylabel("Sepal Width [cm]", fontsize=25)
ax.set_zlabel("Petal Length [cm]", fontsize=25)

# Remove tick labels
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# Add color bar for the scatter plot
#cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
#cbar.set_label("Class Label", fontsize=20)

#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=10)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
accuracy = accuracy_score(y_test, predictions)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')