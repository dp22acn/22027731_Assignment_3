# Importing Nessecery Libraries
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter("ignore", FutureWarning)


def retrieve_emissions_data(indicator):
    """
    Retrieve greenhouse gas emissions data from the World Bank API.

    Args:
        indicator (str): Indicator code for the emissions data.

    Returns:
        pandas.DataFrame: DataFrame containing the emissions data.
    """
    # Construct the API URL
    url = f"http://api.worldbank.org/v2/country/all/indicator/{indicator}"

    # Set the parameters for API request
    params = {
        "date": "1990:2018",
        "format": "json",
        "per_page": "10000"
    }

    # Initialize variables
    data = []
    page = 1

    # Retrieve data from API in paginated form
    while True:
        params["page"] = str(page)
        response = requests.get(url, params=params)

        # Break the loop if the response is not successful
        if response.status_code != 200:
            break

        # Extract data from the response
        page_data = response.json()[1]

        # Break the loop if there is no more data available
        if len(page_data) == 0:
            break

        # Append the page data to the main data list
        data += page_data
        page += 1

    # Convert data to a pandas DataFrame and rename columns
    df_emissions = pd.json_normalize(data).rename(
        columns={'value': name, 'country.value': 'Country Name', 'date': 'Year'
                 }
    )[['Country Name', 'Year', name]]

    # Pivot the DataFrame to have years as columns and fill missing values
    # using forward fill
    df_emissions = df_emissions.pivot(
        index='Country Name', columns='Year', values=name
    ).fillna(method='ffill', axis=1)

    return df_emissions


def visualize_clusters(emissions_reduced_data, cluster_labels,
                       cluster_centers, name):
    """
    Visualize the clustered data using t-SNE dimensions.

    Args:
        emissions_reduced_data (numpy.ndarray): Reduced data after t-SNE
        dimensionality reduction.
        cluster_labels (numpy.ndarray): Cluster labels for each data point.
        cluster_centers (numpy.ndarray): Coordinates of the cluster centers.
        name (str): Name of the emissions data.
    """
    # Plot the clustered data using t-SNE dimensions
    plt.figure(figsize=(8, 6))
    for label in np.unique(cluster_labels):
        plt.scatter(
            *emissions_reduced_data[cluster_labels == label].T,
            label=f'Cluster {label}')

    # Plot the cluster centers
    plt.scatter(*cluster_centers.T, c='red', marker='x', s=100)

    # Set the plot labels and title
    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')
    plt.legend()
    plt.title(f'KMeans Clustering of {name} with 2 dimensions', fontsize=10)

    # Save the plot to a file
    plt.savefig(f'kmeans-{name.replace(" ", "")}.png',
                dpi=600, transparent=True)
    plt.show()
