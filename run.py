# Notes:-
# This code uses CART instead of c4.5. They both are pretty close to each other. Refer documentation for difference.
# This code uses SAMME.R for Adaboost instea of ADaboost.MI used for boosting in the paper.
#
# Links to documentations
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit

# TODO
# Re-verify all parameters to ensembles methods and make sure we are not diverging from the paper.
# Do fancy charts with the accuracy data.
# Find out if there is a way in extratreeclassifier to make it use only positive information gain splits.
# Are we going to use pruning?
# calculate Error rate using t-test
# Support other data source format besides csv



# Import stuff
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.preprocessing import Imputer

CSV_SEP = ','

# get list of datasets of UCI rep to use
def get_dataset_list():
    #datasets = ['https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    #            'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv']
    datasets = ['data/anneal.data.txt','data/glass.data.txt','data/heart.dat.txt','data/house-votes-84.data.txt']
    #,'data/krkopt.data.txt',
    #'data/letter-recognition.data.txt','data/sat.trn.txt','data/segment.dat.txt','data/sonar.all-data.txt',
    #'data/soybean-large.data.txt','splice.data.txt']

    return datasets

# Do bagging.
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_bagging(X, y):
    bagging = BaggingClassifier(DecisionTreeClassifier(), 200, 0.67, 1.0, True, True)
    return cross_val_score(bagging, X, y, cv=10)

# Do boosting
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_boosting(X, y):
    boosting = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100)
    return cross_val_score(boosting, X, y, cv=10)

# Do Randomization
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_randomization(X, y):
    random = ExtraTreesClassifier(200)
    return cross_val_score(random, X, y, cv=10)

# Do plain vanilla CART
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_cart(X, y):
    cart = DecisionTreeClassifier()
    return cross_val_score(cart, X, y, cv=10)

def convert_to_error_rate(score):
    error = []
    for foldscore in score:
        errorscore = 1 - foldscore
        error.append(errorscore)
    return error
    

# Main
scores = []
error_rates = []
mean_error_rates = []
for dataset_url in get_dataset_list():
    print("Testing: "+dataset_url)
    datascore = []
    dataerrorrate = []
    df = pd.read_csv(dataset_url, CSV_SEP)
    Y = df['class'].values
    df = df.drop('class', 1)
    df.fillna(df.mode().iloc[0])
    X = df.as_matrix()
    print(X)
    #imp = Imputer(missing_values='?', strategy='mode',axis=0)
    #X = imp.fit_transform(df)
    Y = np.array([1 if y >= 7 else 0 for y in Y])
    
    cart_score = do_cart(X, Y)
    bagging_score = do_bagging(X, Y)
    boosting_score = do_boosting(X, Y)
    random_score = do_randomization(X, Y)
    
    datascore.append(cart_score)
    datascore.append(bagging_score)
    datascore.append(boosting_score)
    datascore.append(random_score)
    scores.append(datascore)
    
    cart_error = convert_to_error_rate(cart_score)
    bagging_error = convert_to_error_rate(bagging_score)
    boosting_error = convert_to_error_rate(boosting_score)
    random_error = convert_to_error_rate(random_score)
    
    dataerrorrate.append(cart_error)
    dataerrorrate.append(bagging_error)
    dataerrorrate.append(boosting_error)
    dataerrorrate.append(random_error)
    error_rates.append(dataerrorrate)
    
    mean_error_rates.append([np.mean(cart_error), np.mean(bagging_error), np.mean(boosting_error), np.mean(random_error)])


# Plot box plot.
plt.boxplot(scores)
plt.ylabel('Score')
plt.xlabel('Classification data set')
plt.title('Box plot of classification score for various datasets')
plt.show()

# Plot box plot.
plt.boxplot(error_rates)
plt.ylabel('Error rates')
plt.xlabel('Classification data set')
plt.title('Box plot of error rates for various datasets')
plt.show()

# Plot error rates of various algorithms
mean_error_rates = np.array(mean_error_rates)
plt.plot(mean_error_rates.T[0], label='CART')
plt.plot(mean_error_rates.T[1], label='Bagging')
plt.plot(mean_error_rates.T[2], label='Boosting')
plt.plot(mean_error_rates.T[3], label='Randomization')
plt.legend(loc='best')
plt.ylabel('Error rates')
plt.xlabel('Classification data set')
plt.title('Line plot of error rates for various datasets using various algorithms')
plt.show()

print(scores)