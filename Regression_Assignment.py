#import relevent packages, including sklearn
from sklearn.datasets import load_boston
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

#load data set
boston = load_boston()

#identify dataset shape and columns
print(boston.data.shape)
print(boston.feature_names)
print(boston.DESCR)

#Select features that may have linear relationship to price = CRIM, RAD, PTRATIO, and AGE

#Data set is turned into a data frame
boston_df = pd.DataFrame(boston.data)

#add column names to data set from features
boston_df.columns = boston.feature_names

#add price to the data frame
boston_df['PRICE'] = boston.target

#divide out price
X = boston_df.drop('PRICE', axis=1)
Y = boston_df['PRICE']

#Divide into training and testing set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = .3, random_state = 3)

#Fit the model
reg = LinearRegression()
reg.fit(X_train, Y_train)
fit = reg.fit(X_train, Y_train)

#predict
pred = reg.predict(X_test)

#plot results
plt.scatter(Y_test, pred)
plt.xlabel("Actual Prices (in thousands)")
plt.ylabel("Predicted prices (in thousands)")
plt.title("Linear Regression: Actual vs. Predicted Prices")
plt.show()






