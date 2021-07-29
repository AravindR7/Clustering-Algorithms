"""New Project Example

This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. Every ML model project
should have a definition file like this one.

"""
from typing import Any

import layer
from layer import Featureset, Train
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def train_model(train: Train, pf: Featureset("customer_features")) -> Any:
    """Model train function

    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.

    Args:
        train (layer.Train): Represents the current train of the model, passed by
            Layer when the training of the model starts.
        pf (spark.DataFrame): Layer will return all features inside the
            `features` featureset as a spark.DataFrame automatically
            joining them by primary keys, described in the dataset.yaml

    Returns:
       model: Trained model object

    """
    data = pf.to_pandas()

    n_clusters = 4
    train.log_parameter("n_clusters", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters)
    fit_data = kmeans.fit(data)

    label = fit_data.predict(data)
    # Calculating Silhouette score
    sc = silhouette_score(data, label)
    train.log_metric("SilhouetteScore", sc)

    return fit_data
