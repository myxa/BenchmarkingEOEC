from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from nilearn.connectome import sym_matrix_to_vec
from sklearn.metrics import f1_score


class LogRegPCA:
    def __init__(self, pca=True):
        """
        Initialize a LogRegPCA object.

        Parameters
        ----------
        pca : bool, default=True
            If True, a PCA object is created and used for dimensionality
            reduction. If False, no dimensionality reduction is performed.

        """
        self.pca = PCA() if pca else None
        self.model = LogisticRegression()
    
    def model_training(self, x, y):
        """
        Train a logistic regression model using the given data.

        Parameters
        ----------
        x : array-like
            The data to be used for training the model.
        y : array-like
            The labels corresponding to the data.

        Returns
        -------
        accuracy : float
            The accuracy of the model on the training data.
        """
        vecs = sym_matrix_to_vec(x, discard_diagonal=True)

        if self.pca is not None:
            vecs = self.pca.fit_transform(vecs)

        self.model.fit(vecs, y)
        acc = self.model.score(vecs, y)

        print('Accuracy on train:', round(acc, 3))

        return acc
    
    def model_testing(self, x, y):
        """
        Test the model on the given data.

        Parameters
        ----------
        x : array-like
            The data to be used for testing the model.
        y : array-like
            The labels corresponding to the data.

        Returns
        -------
        cm : array-like
            The confusion matrix corresponding to the test data.
        accuracy : float
            The accuracy of the model on the test data.
        f1 : float
            The F1 score of the model on the test data.
        """
        self.vecs = sym_matrix_to_vec(x, discard_diagonal=True) #self.preprocess(x)

        if self.pca is not None:
            self.vecs = self.pca.transform(self.vecs)

        y_pred = self.model.predict(self.vecs)
        
        acc = self.model.score(self.vecs, y)
        f1 = f1_score(y, y_pred)
        print('Accuracy on test:', round(acc, 3))
        print('F1 score on test:', round(f1, 3))
        cm = confusion_matrix(y, y_pred)

        return cm, acc, f1
    

def cross_validation(model, x, y, k=5):
    """
    Perform k-fold cross-validation on the given data.

    Parameters
    ----------
    model : object
        The model to be used for cross-validation.
    x : array-like
        The data to be used for cross-validation.
    y : array-like
        The labels corresponding to the data.
    k : int, default=5
        The number of folds to be used for cross-validation.

    Returns
    -------
    accuracy : float
    """
    pass
