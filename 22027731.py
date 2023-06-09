# -- coding: utf-8 --
"""
Created on Fri May 12 17:18:01 2023

@author: DEVENDRA PEDDOJU
"""
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


# Set the indicators and countries you want to retrieve data for
indicator = 'EN.ATM.GHGT.KT.CE'
name = 'Total greenhouse gas emissions'

# Retrieve emissions data
df_emissions = retrieve_emissions_data(indicator)

# Scale the emissions using MinMaxScaler
emissions_scaled = MinMaxScaler().fit_transform(df_emissions.fillna(0))

# Perform t-SNE dimensionality reduction on scaled emissions data
emissions_reduced_data = TSNE(n_components=2).fit_transform(emissions_scaled)

# Perform K-means clustering on the reduced data
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=2023)
cluster_labels = kmeans.fit_predict(emissions_reduced_data)
cluster_centers = kmeans.cluster_centers_

# Add cluster labels to the emissions DataFrame
df_emissions['Cluster'] = cluster_labels

# Visualize the clustered data using t-SNE dimensions
visualize_clusters(emissions_reduced_data,
                   cluster_labels, cluster_centers, name)

# Filter the DataFrame for each cluster
cluster_0 = df_emissions[df_emissions['Cluster'] == 0]
cluster_1 = df_emissions[df_emissions['Cluster'] == 1]
cluster_2 = df_emissions[df_emissions['Cluster'] == 2]

# Fit curve and visualize emissions trend for China in cluster 1
xnew_china, ynew_china = fit_curve_and_visualize(
    df_emissions, cluster_labels, 1, "China")

# Fit curve and visualize emissions trend for Germany in cluster 1
xnew_germany, ynew_germany = fit_curve_and_visualize(
    df_emissions, cluster_labels, 1, "Germany", beta=16)


def reshape_and_plot_global_trend(df_emissions):
    """
    Reshape the emissions DataFrame and plot the global emissions trend.

    Args:
        df_emissions (pandas.DataFrame): DataFrame containing
        the emissions data.
    """
    # Reshape the DataFrame using melt function
    df_orig = pd.melt(df_emissions.reset_index(),
                      id_vars=['Country Name', 'Cluster'],
                      value_vars=list(df_emissions.columns), var_name='Year',
                      value_name='Total Greenhouse Gas Emissions')

    # Calculate the global greenhouse gas emissions trend by year
    global_co2_trend = df_orig.groupby(
        'Year')['Total Greenhouse Gas Emissions'].sum()

    # Plot the global greenhouse gas emissions trend
    plt.plot(global_co2_trend.index, global_co2_trend.values)
    plt.gca().set_xticks(plt.gca().get_xticks()[::5])
    plt.xlabel('Year')
    plt.ylabel('Greenhouse Gas Emissions')
    plt.title('Global Greenhouse Gas Emissions Trend')

    # Save the plot as an image file
    plt.savefig('Global.png', dpi=300, transparent=True)
    plt.show()


def calculate_cluster_statistics(cluster_labels, df_emissions):
    """
    Calculate cluster statistics.

    Args:
        cluster_labels (numpy.ndarray): Cluster labels for each data point.
        df_emissions (pandas.DataFrame): DataFrame containing the
        emissions data.

    Returns:
        pandas.DataFrame: DataFrame containing cluster statistics.
    """
    # Compute cluster sizes
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

    # Compute variance of emissions within each cluster
    variance = pd.DataFrame(df_emissions).groupby(cluster_labels).var()
    variance_means = variance.mean(axis=1)

    # Create a tabular representation of cluster statistics
    cluster_stats = pd.DataFrame({
        'Cluster': range(kmeans.n_clusters),
        'Size': cluster_sizes,
        'Centroid': cluster_centers.tolist(),
        'Variance': variance_means.tolist(),
    })
    cluster_stats = cluster_stats.set_index('Cluster')

    return cluster_stats


# Reshape the DataFrame and plot the global emissions trend
reshape_and_plot_global_trend(df_emissions)

# Calculate cluster statistics
cluster_stats = calculate_cluster_statistics(cluster_labels, df_emissions)

# Save cluster statistics to a CSV file
cluster_stats.to_csv("stats.csv")
