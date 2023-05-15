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
