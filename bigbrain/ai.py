# Artificial Intelligence is finally here

import random
import numpy as np

# data modules
from sklearn.model_selection import train_test_split

# kernel modules
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic

# classification modules
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from mvlearn.construct import random_subspace_method
try:
    from mvlearn.semi_supervised import CTClassifier
except:
    from mvlearn.cotraining import CTClassifier

# regression modules
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso

# evaluation modules
from sklearn.metrics import accuracy_score, mean_squared_error
from numpy import inf
from sklearn.model_selection import GridSearchCV


class AI():
    """
    A class for learning anything from data.

    Parameters
    ----------

    Attributes
    ----------
    """
    def __init__(self, model_type="classification", test_size=None, seed=None, verbose=False):
        self.model_ = None
        self.model_type_ = model_type
        self.test_size_ = test_size
        self.seed_ = seed
        self.verbose_ = verbose
        self.extra_info_ = {'random_state': None}

    def learn(self, X, y):
        """
        Learn the best model for the data.

        Parameters
        ----------
        X : nd-array
            Data array (n_samples, n_features)
        y : nd-array
            Targets.

        Returns
        -------
        """

        # split into train/validation (default 75/25)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size_, random_state=self.seed_)

        # loop through list of supervised learning classification methods
        if self.model_type_ == "classification":
            models = [MLPClassifier(alpha=1, max_iter=1000),
                KNeighborsClassifier(),
                SVC(kernel="linear", C=0.025),
                SVC(kernel="poly", C=1),
                SVC(kernel="rbf", gamma=2, C=1),
                SVC(kernel="sigmoid", C=1),
                GaussianProcessClassifier(RBF()),
                GaussianProcessClassifier(ConstantKernel()),
                GaussianProcessClassifier(RationalQuadratic()),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                AdaBoostClassifier(),
                GaussianNB(),
                LinearDiscriminantAnalysis(),
                QuadraticDiscriminantAnalysis()]

            if len(np.unique(y)) == 2:
                models.append(CTClassifier())

            best_score = 0

        else:
            models = [MLPRegressor(alpha=1, max_iter=1000),
                KNeighborsRegressor(),
                SVR(kernel="linear", C=0.025),
                SVR(kernel="poly", C=1),
                SVR(kernel="rbf", gamma=2, C=1),
                SVR(kernel="sigmoid", C=1),
                GaussianProcessRegressor(RBF()),
                GaussianProcessRegressor(ConstantKernel()),
                GaussianProcessRegressor(RationalQuadratic()),
                DecisionTreeRegressor(max_depth=5),
                RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1),
                AdaBoostRegressor(),
                Lasso()]

            best_score = inf

        # for each model, fit on training data then predict on testing data

        for m in models:

            state_int = None

            # check for multiview model
            if hasattr(m, 'estimator1_') or hasattr(m, 'estimator1'):
                state_int = np.random.randint(low=0, high=1e5)

                np.random.seed(state_int)
                random.seed(state_int)
                Xs_train = random_subspace_method(X_train, n_features=0.7, n_views=2)
                m.fit(Xs_train, y_train)

                np.random.seed(state_int)
                random.seed(state_int)
                Xs_test = random_subspace_method(X_test, n_features=0.7, n_views=2)
                y_pred = m.predict(Xs_test)
            else:    
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)

            if self.model_type_ == "classification":
                score = accuracy_score(y_test, y_pred)

                # if testing accuracy is the best so far
                if score > best_score:
                    # set it as the best score and best model
                    best_score = score
                    self.model_ = m
                    self.extra_info_['random_state'] = state_int
                score_name = "accuracy score"
            else:
                score = mean_squared_error(y_test, y_pred)

                # if testing accuracy is the best so far
                if score < best_score:
                    # set it as the best score and best model
                    best_score = score
                    self.model_ = m
                    self.extra_info_['random_state'] = state_int
                score_name = "MSE"

        # grid search parameterss for best model and save

        # done learning
        print("My big brain has learned everything.\n")

        if self.verbose_:
            print("Best model: %s" % self.model_)
            print("Best %s: %0.3f" % (score_name, best_score))

    def go(self, X):
        """
        Learn the best model for the data.

        Parameters
        ----------
        X : nd-array
            Data array (n_samples, n_features)

        Returns
        -------
        y_pred : nd-array
            Predicted targets.
        """

        if hasattr(self.model_, 'estimator1_') or hasattr(self.model_, 'estimator1'):
            np.random.seed(self.extra_info_['random_state'])
            random.seed(self.extra_info_['random_state'])
            Xs = random_subspace_method(X, n_features=0.7, n_views=2)
            return self.model_.predict(Xs)

        else:
            return self.model_.predict(X)
