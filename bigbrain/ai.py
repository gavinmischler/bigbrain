# Artificial Intelligence is finally here

import sklearn


class AI():
    """
    A class for learning anything from data.

    Parameters
    ----------

    Attributes
    ----------
    """
    def __init__(self):
        self.model_ = None

    def learn(X, y):
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
        fitted_model : returns an instance of self
        """

        # split into 75/25 train/validation
        # loop through list of supervised learning methods
        # and for each one, fit on train_set then predict
        # on validation set, and if its validation accuracy
        # is the best so far, set it as self.model_
        # Then grid search params for this model and save
        # When done searching, print something like "Learned everything"

    def go(X):
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
