from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

basedir = os.path.dirname(__file__)

# importing data from csv file
def main():
    data = pd.read_csv(os.path.join(basedir, 'files', 'FinalTransformedData_ReadyForClassification_v2.csv'))
    data.drop(['COVID_MONTH'], axis=1, inplace=True)
    # Select the features to use for clustering
    X = data[['AIRPORT_SPIKE', 'COVID_SPIKE']]
    X.head()
    le = LabelEncoder()
    X['AIRPORT_NAME'] = le.fit_transform(data['AIRPORT_NAME'])
    X['STATE'] = le.fit_transform(data['STATE'])
    X['COUNTRY'] = le.fit_transform(data['COUNTRY'])

    # Standardize the features to have zero mean and unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    # # using Kmeans

    # Create a KMeans object with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, max_iter=1000)

    # Fit the model to the data
    kmeans.fit(X)

    # Predict the cluster labels for each data point
    labels = kmeans.predict(X)

    data['CLUSTER'] = kmeans.labels_

    data.head()

    x_orig = scaler.inverse_transform(X)

    label_names = ['Cluster 1', 'Cluster 2', 'Cluster 3']

    for i in range(len(label_names)):
        plt.scatter(data.iloc[data['CLUSTER'].values == i, 4], data.iloc[data['CLUSTER'].values == i, 5], label=label_names[i])

    plt.xlabel('AIRPORT FLIGHT NUMBERS')
    plt.ylabel('COVID NUMBERS')
    plt.legend()
    plt.show()

    cluster_centers = kmeans.cluster_centers_

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

    colors = ['black', 'purple', 'pink']
    label_names = ['Cluster 1', 'Cluster 2', 'Cluster 3']

    for i, center in enumerate(cluster_centers):
        plt.scatter(center[0], center[1], c=colors[i], s=400, alpha=0.5, label=label_names[i])

    plt.xlabel('AIRPORT FLIGHT NUMBERS')
    plt.ylabel('COVID NUMBERS')
    plt.legend()
    plt.show()

    colors = ['red', 'green', 'blue', 'purple', 'orange']

    plt.scatter(data['MONTH_x'], data['AIRPORT_SPIKE'],label=data['AIRPORT_NAME'])
    plt.xticks(rotation=90)
    plt.xlabel('Month')
    plt.ylabel('Airport Spike')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.show()

    # Use the pivot method to reshape the data into a 2D table
    data_pivot = data.pivot(index='AIRPORT_NAME', columns='MONTH_x', values='AIRPORT_SPIKE')

    # Create the heatmap using the pcolor method
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data_pivot, cmap='YlGnBu')

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the x-axis tick labels to the month names
    ax.set_xticks(np.arange(len(data['MONTH_x'].unique())) + 0.5, minor=False)
    ax.set_xticklabels(data['MONTH_x'].unique(), minor=False, rotation=90)

    # Set the y-axis tick labels to the airport names
    ax.set_yticks(np.arange(len(data['AIRPORT_NAME'].unique())) + 0.5, minor=False)
    ax.set_yticklabels(data['AIRPORT_NAME'].unique(), minor=False, fontsize=8, rotation=30)

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Airport Name')
    plt.title('Airport Values by Month')

    plt.show()


    filtered_data_ga = data[data['STATE'] == 'GEORGIA']

    # Use the pivot method to reshape the data into a 2D table
    data_pivot_ga = filtered_data_ga.pivot(index='AIRPORT_NAME', columns='MONTH_x', values='AIRPORT_SPIKE')
    # Create the heatmap using the pcolor method
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data_pivot_ga, cmap='RdYlGn_r')

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the x-axis tick labels to the month names
    ax.set_xticks(np.arange(len(filtered_data_ga['MONTH_x'].unique())) + 0.5, minor=False)
    ax.set_xticklabels(filtered_data_ga['MONTH_x'].unique(), minor=False, rotation=90)

    # Set the y-axis tick labels to the airport names
    ax.set_yticks(np.arange(len(filtered_data_ga['AIRPORT_NAME'].unique())) + 0.5, minor=False)
    ax.set_yticklabels(filtered_data_ga['AIRPORT_NAME'].unique(), minor=False, fontsize=8, rotation=30)

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Airport Name')
    plt.title('Airport Values by Month')

    plt.show()


    data_pivot_ga = filtered_data_ga.pivot(index='AIRPORT_NAME', columns='MONTH_x', values='COVID_SPIKE')
    # Create the heatmap using the pcolor method
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data_pivot_ga, cmap='RdYlGn_r')

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the x-axis tick labels to the month names
    ax.set_xticks(np.arange(len(filtered_data_ga['MONTH_x'].unique())) + 0.5, minor=False)
    ax.set_xticklabels(filtered_data_ga['MONTH_x'].unique(), minor=False, rotation=90)

    # Set the y-axis tick labels to the airport names
    ax.set_yticks(np.arange(len(filtered_data_ga['AIRPORT_NAME'].unique())) + 0.5, minor=False)
    ax.set_yticklabels(filtered_data_ga['AIRPORT_NAME'].unique(), minor=False, fontsize=8, rotation=30)

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Airport Name')
    plt.title('Airport Values by Month')

    plt.show()


    # georgia_data = data[data['STATE'] == 'GEORGIA']
    filtered_data_ga_array = filtered_data_ga.values
    plt.scatter(filtered_data_ga_array[:, 4].astype(int), filtered_data_ga_array[:, 5].astype(int), c=filtered_data_ga_array[:, 6], s=50, cmap='viridis')
    colors = ['black', 'purple', 'pink']
    label_names = ['Cluster 1', 'Cluster 2', 'Cluster 3']

    for i, center in enumerate(cluster_centers):
        plt.scatter(center[0], center[1], c=colors[i], s=400, alpha=0.5, label=label_names[i])

    plt.xlabel('AIRPORT FLIGHT NUMBERS')
    plt.ylabel('COVID NUMBERS')
    plt.legend()
    plt.show()


    # # New York
    #
    # Airport Flight Data

    filtered_data_ny = data[data['STATE'] == 'NEW YORK']

    # Use the pivot method to reshape the data into a 2D table
    data_pivot_ny = filtered_data_ny.pivot(index='AIRPORT_NAME', columns='MONTH_x', values='AIRPORT_SPIKE')
    # Create the heatmap using the pcolor method
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data_pivot_ny, cmap='RdYlGn_r')

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the x-axis tick labels to the month names
    ax.set_xticks(np.arange(len(filtered_data_ny['MONTH_x'].unique())) + 0.5, minor=False)
    ax.set_xticklabels(filtered_data_ny['MONTH_x'].unique(), minor=False, rotation=90)

    # Set the y-axis tick labels to the airport names
    ax.set_yticks(np.arange(len(filtered_data_ny['AIRPORT_NAME'].unique())) + 0.5, minor=False)
    ax.set_yticklabels(filtered_data_ny['AIRPORT_NAME'].unique(), minor=False, fontsize=8, rotation=30)

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Airport Name')
    plt.title('Airport Values by Month')

    plt.show()


    # # New York
    #
    # Covid numbers

    # In[224]:


    filtered_data_ny = data[data['STATE'] == 'NEW YORK']

    # Use the pivot method to reshape the data into a 2D table
    data_pivot_ny = filtered_data_ny.pivot(index='AIRPORT_NAME', columns='MONTH_x', values='COVID_SPIKE')
    # Create the heatmap using the pcolor method
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data_pivot_ny, cmap='RdYlGn_r')

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the x-axis tick labels to the month names
    ax.set_xticks(np.arange(len(filtered_data_ny['MONTH_x'].unique())) + 0.5, minor=False)
    ax.set_xticklabels(filtered_data_ny['MONTH_x'].unique(), minor=False, rotation=90)

    # Set the y-axis tick labels to the airport names
    ax.set_yticks(np.arange(len(filtered_data_ny['AIRPORT_NAME'].unique())) + 0.5, minor=False)
    ax.set_yticklabels(filtered_data_ny['AIRPORT_NAME'].unique(), minor=False, fontsize=8, rotation=30)

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Airport Name')
    plt.title('Airport Values by Month')

    plt.show()


    # # New York Data with Clusters

    filtered_data_ny_array = filtered_data_ny.values
    plt.scatter(filtered_data_ny_array[:, 4].astype(int), filtered_data_ny_array[:, 5].astype(int), c=filtered_data_ny_array[:, 6], s=50, cmap='viridis')
    colors = ['black', 'purple', 'pink']
    label_names = ['Cluster 1', 'Cluster 2', 'Cluster 3']

    for i, center in enumerate(cluster_centers):
        plt.scatter(center[0], center[1], c=colors[i], s=400, alpha=0.5, label=label_names[i])

    plt.xlabel('AIRPORT FLIGHT NUMBERS')
    plt.ylabel('COVID NUMBERS')
    plt.legend()
    plt.show()


    # # California Airport Data

    # In[226]:


    filtered_data_ca = data[data['STATE'] == 'CALIFORNIA']

    # Use the pivot method to reshape the data into a 2D table
    data_pivot_ca = filtered_data_ca.pivot(index='AIRPORT_NAME', columns='MONTH_x', values='AIRPORT_SPIKE')
    # Create the heatmap using the pcolor method
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data_pivot_ca, cmap='RdYlGn_r')

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the x-axis tick labels to the month names
    ax.set_xticks(np.arange(len(filtered_data_ca['MONTH_x'].unique())) + 0.5, minor=False)
    ax.set_xticklabels(filtered_data_ca['MONTH_x'].unique(), minor=False, rotation=90)

    # Set the y-axis tick labels to the airport names
    ax.set_yticks(np.arange(len(filtered_data_ca['AIRPORT_NAME'].unique())) + 0.5, minor=False)
    ax.set_yticklabels(filtered_data_ca['AIRPORT_NAME'].unique(), minor=False, fontsize=8, rotation=30)

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Airport Name')
    plt.title('Airport Values by Month')

    plt.show()


    # # California Covid Data

    filtered_data_ca = data[data['STATE'] == 'CALIFORNIA']

    # Use the pivot method to reshape the data into a 2D table
    data_pivot_ca = filtered_data_ca.pivot(index='AIRPORT_NAME', columns='MONTH_x', values='COVID_SPIKE')
    # Create the heatmap using the pcolor method
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data_pivot_ca, cmap='RdYlGn_r')

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the x-axis tick labels to the month names
    ax.set_xticks(np.arange(len(filtered_data_ca['MONTH_x'].unique())) + 0.5, minor=False)
    ax.set_xticklabels(filtered_data_ca['MONTH_x'].unique(), minor=False, rotation=90)

    # Set the y-axis tick labels to the airport names
    ax.set_yticks(np.arange(len(filtered_data_ca['AIRPORT_NAME'].unique())) + 0.5, minor=False)
    ax.set_yticklabels(filtered_data_ca['AIRPORT_NAME'].unique(), minor=False, fontsize=8, rotation=30)

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Airport Name')
    plt.title('Airport Values by Month')

    plt.show()


    # # California Clustered Data

    filtered_data_ca_array = filtered_data_ca.values
    plt.scatter(filtered_data_ca_array[:, 4].astype(int), filtered_data_ca_array[:, 5].astype(int), c=filtered_data_ca_array[:, 6], s=50, cmap='viridis')
    colors = ['black', 'purple', 'pink']
    label_names = ['Cluster 1', 'Cluster 2', 'Cluster 3']

    for i, center in enumerate(cluster_centers):
        plt.scatter(center[0], center[1], c=colors[i], s=400, alpha=0.5, label=label_names[i])

    plt.xlabel('AIRPORT FLIGHT NUMBERS')
    plt.ylabel('COVID NUMBERS')
    plt.legend()
    plt.show()


    # # Texas Airport Data

    filtered_data_tx = data[data['STATE'] == 'TEXAS']

    # Use the pivot method to reshape the data into a 2D table
    data_pivot_tx = filtered_data_tx.pivot(index='AIRPORT_NAME', columns='MONTH_x', values='AIRPORT_SPIKE')
    # Create the heatmap using the pcolor method
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data_pivot_tx, cmap='RdYlGn_r')

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the x-axis tick labels to the month names
    ax.set_xticks(np.arange(len(filtered_data_tx['MONTH_x'].unique())) + 0.5, minor=False)
    ax.set_xticklabels(filtered_data_tx['MONTH_x'].unique(), minor=False, rotation=90)

    # Set the y-axis tick labels to the airport names
    ax.set_yticks(np.arange(len(filtered_data_tx['AIRPORT_NAME'].unique())) + 0.5, minor=False)
    ax.set_yticklabels(filtered_data_tx['AIRPORT_NAME'].unique(), minor=False, fontsize=8, rotation=30)

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Airport Name')
    plt.title('Airport Values by Month')

    plt.show()


    # # Texas Covid Data

    filtered_data_tx = data[data['STATE'] == 'TEXAS']

    # Use the pivot method to reshape the data into a 2D table
    data_pivot_tx = filtered_data_tx.pivot(index='AIRPORT_NAME', columns='MONTH_x', values='COVID_SPIKE')
    # Create the heatmap using the pcolor method
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data_pivot_tx, cmap='RdYlGn_r')

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Set the x-axis tick labels to the month names
    ax.set_xticks(np.arange(len(filtered_data_tx['MONTH_x'].unique())) + 0.5, minor=False)
    ax.set_xticklabels(filtered_data_tx['MONTH_x'].unique(), minor=False, rotation=90)

    # Set the y-axis tick labels to the airport names
    ax.set_yticks(np.arange(len(filtered_data_tx['AIRPORT_NAME'].unique())) + 0.5, minor=False)
    ax.set_yticklabels(filtered_data_tx['AIRPORT_NAME'].unique(), minor=False, fontsize=8, rotation=30)

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Airport Name')
    plt.title('Airport Values by Month')

    plt.show()


    # # Texas Data Clustered

    filtered_data_tx_array = filtered_data_tx.values
    plt.scatter(filtered_data_tx_array[:, 4].astype(int), filtered_data_tx_array[:, 5].astype(int), c=filtered_data_tx_array[:, 6], s=50, cmap='viridis')
    colors = ['black', 'purple', 'pink']
    label_names = ['Cluster 1', 'Cluster 2', 'Cluster 3']

    for i, center in enumerate(cluster_centers):
        plt.scatter(center[0], center[1], c=colors[i], s=400, alpha=0.5, label=label_names[i])

    plt.xlabel('AIRPORT FLIGHT NUMBERS')
    plt.ylabel('COVID NUMBERS')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()