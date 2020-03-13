# Artificial Intelligence is finally here

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
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
            models = [KNeighborsClassifier(3),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1),GaussianProcessClassifier(1.0 * RBF(1.0)),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), 
                MLPClassifier(alpha=1, max_iter=1000),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis()]

        # for each model, fit on training data then predict on testing data
        best_score = 0

        for m in models:
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            # if testing accuracy is the best so far
            if score > best_score:
                # set it as the best score and best model
                best_score = score
                self.model_ = m

        # grid search parameterss for best model and save

        # done learning
        print("My big brain has learned everything.")

        if self.verbose_:
            print("Best model: %s" % self.model_)
            print("Best model score: %0.3f" % best_score)

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

        return self.model_.predict(X)
