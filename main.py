# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#source: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape
print("rows + Columns", dataset.shape)
# head
print(dataset.head(20))
# see summary of each attribute 
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

# data visualisations 
# univariate plots - shows the data and summarizes its distribution
# to better understand each attribute 
# multivariate plots - better understand the relationship between attributes 


# univariate plot - box and whisker
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# univariate plot - histogram
dataset.hist()
pyplot.show()

# multivariate plot - scatterplot
scatter_matrix(dataset)
pyplot.show()

# create a validation dataset, to verify that the model is all good
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
