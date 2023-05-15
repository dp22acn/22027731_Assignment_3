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


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    This routine can be used in assignment programs.
    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


def fit_curve_and_visualize(df_emissions, cluster_labels,
                            cluster_num, country, beta=1):
    """
    Fit a curve to the emissions data of a specific country in a cluster
    and visualize it.

    Args:
        df_emissions (pandas.DataFrame): DataFrame containing
        the emissions data.
        cluster_labels (numpy.ndarray): Cluster labels for each data point.
        cluster_num (int): Cluster number to consider.
        country (str): Name of the country.

    Returns:
        numpy.ndarray: Fitted curve x values.
        numpy.ndarray: Fitted curve y values.
    """
    def exponential_growth(x, a, b, c):
        """
   Exponential growth function for curve fitting.

   Args:
       x (numpy.ndarray): Independent variable.
       a (float): Coefficient for linear term.
       b (float): Coefficient for quadratic term.
       c (float): Coefficient for constant term.

   Returns:
       numpy.ndarray: Predicted values based
       on the exponential growth function.
   """
        return a * x + b * x ** 2 + c

    # Extract x and y data for fitting curve in the specified
    # cluster and country

    xdata = df_emissions.loc[country][:-1].index.values.astype('int')
    ydata = df_emissions.loc[country][:-1].values

    # Perform curve fitting using exponential_growth function

    popt, pcov = curve_fit(exponential_growth, xdata, ydata)

    # Generate new x values for smooth curve visualization
    xnew = np.arange(min(xdata), max(xdata) + 5, 1)
    ynew = exponential_growth(xnew, *popt)
    lower, upper = err_ranges(xnew, exponential_growth, popt,
                              [0.1, 0.01, 0.001])

    # Plot the best fitting function, data points, and labels
    plt.plot(xnew, ynew, 'r-', label='Best fitting function')
    plt.scatter(xdata, ydata, label='Data')
    plt.fill_between(xnew, ynew-lower[0]/beta, ynew+upper[0]/beta,
                     alpha=0.2, label='Confidence Interval')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Greenhouse Gas emissions')
    plt.title(f'Greenhouse Gas emissions trend for {country}')

    # Save the plot as an image file
    plt.savefig(f'{country}.png', dpi=300, transparent=True)
    plt.show()

    return xnew, ynew
