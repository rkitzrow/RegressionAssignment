# Import sklearn and data set for iris
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# load data set
iris = load_iris()

# Identify dataset shape and columns

# Data has 150 rows and 4 columns
#print(iris.data.shape)

# Feature names are ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#print(iris.feature_names)
colnames = ['sepal_length','sepal_width','petal_length','petal_width']

# Desc recommends 3 classes with 50 belonging to each class (50*3 = 150 rows)
#print(iris.DESCR)

# Get only the data and apply modified column names
iris_df = pd.DataFrame(iris.data)
iris_df.columns = colnames

# Perform kmeans on with multiple clusters, set max iterations to 1000

models = {}

# using this range will give me 10 clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=5000).fit(iris_df)
    models[i] = kmeans.inertia_

# Keys are the clusters (1:10) taken from my range above
clusters = models.keys()

# Values are the inertia for each i cluster I collected above
inertia = models.values()

# I plot clusters (x axis) by the sum of squared distances (y axis)
plt.plot(clusters,inertia)

#I set the title of the plot, x and y axis labels, and show it
plt.title("Inertia by Cluster for IRIS Dataset")
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of squared distances")
plt.show()

#I turn these into a dataframe to view as a table
comb = pd.DataFrame({'Clusters':list(models.values()),'Intertia':list(models.keys())})
print(comb)

# Summary
# The largest stabilization happens between clusters 1-3 (going form 681 to 57)
# From this table the optimal cluster is probably in the 3-4 range.

# Sources:
#1. https://machinelearningmastery.com/machine-learning-in-python-step-by-step/ for looping info
#2. https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion for looping and elbow graphs
#3. https://cmdlinetips.com/2018/01/how-to-create-pandas-dataframe-from-multiple-lists/ for using dictionaries
#4. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for understanding KMeans

