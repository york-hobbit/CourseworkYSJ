# -*- coding: utf-8 -*-
"""
Created on Sat May 28 15:42:52 2022

@author: PWade
"""

# import libraries
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import date
import seaborn as sns

# read data
data = pd.read_csv("marketing_campaign.csv", sep="\t")
data.head()


# Find informtion about the data
data.info()

# Check for unique values
unique = data.nunique()
unique
# Remove un-needed fields ID, Z-CostContact and Z_Revenue
data = data.drop(["ID", "Z_CostContact", "Z_Revenue"], axis=1)

# Calculate age of customer from date of birth
data["Age"] = 2022 - data["Year_Birth"]

# Remove Year of birth
data = data.drop(["Year_Birth"], axis=1)

# Remove income with nan values
data = data.dropna()

# Get values of the different education levels
education_values = data["Education"].unique()
education_values

# Get values of thre different marital statuses
status_values = data["Marital_Status"].unique()
status_values

#df['column name'] = df['column name'].replace(['1st old value','2nd old value',...],['1st new value','2nd new value',...])
# Replace education values with numbers
data["Education"] = data["Education"].replace(
    ["Graduation", "PhD", "Master", "Basic", "2n Cycle"], ["2", "3", "3", "1", "3"])

# Change column type to integer
data["Education"] = data["Education"].astype(int)

# Replace Status values with numbers
data["Marital_Status"] = data["Marital_Status"].replace(
    ["Single", "Together", "Married", "Divorced", "Widow", "Alone", "Absurd", "YOLO"], ["1", "2", "2", "1", "1", "1", "1", "1"])

# Change column type to integer
data["Marital_Status"] = data["Marital_Status"].astype(int)

# Create a single column for Household Marital_Status + Kidhome + Teenhome
data["Household"] = data["Marital_Status"] + data["Kidhome"] + data["Teenhome"]

# Remove Marital_Status, Kidhome, Teenhome
data = data.drop(["Marital_Status", "Kidhome", "Teenhome"], axis=1)

# Calculate the number of days that the member has been part of scheme
today = date.today()
total = pd.to_datetime(today) - pd.to_datetime(data["Dt_Customer"])
data["Member_since"] = total.dt.days

# Remove Dt_Customer
data = data.drop(["Dt_Customer"], axis=1)

# Create a Correlation plot to viualise the variables
sns.pairplot(data, kind="scatter")
plt.show()

# create total number offers accepted
data["TotalCmp"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + \
    data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"]

# Remove offers accepted columns
data = data.drop(["AcceptedCmp1", "AcceptedCmp2",
                 "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"], axis=1)

# As Clustering sensitive to outliers check Income, Age, Household, Membersince and TotalCmp for outliers
sns.set_color_codes("bright")
sns.boxplot(x=data["Income"], color=("m"))
sns.boxplot(x=data["Age"], color=("r"))
sns.boxplot(x=data["Household"], color=("g"))
sns.boxplot(x=data["Member_since"], color=("b"))
sns.boxplot(x=data["TotalCmp"], color=("y"))

# Check numbers for outliers
print("show number of customers with income above 200000: ",
      len(data[data["Income"] > 200000]))
print("show number of customers with age above 90: ",
      len(data[data["Age"] > 90]))
print("show number of customers with household above 4: ",
      len(data[data["Household"] > 4]))

# get count number of rows in dataframe
len(data.index)

# Remove the outliers for income and check to see if removed
data.drop(data[data.Income > 200000].index, inplace=True)
len(data.index)

# Remove outliers for Age and check to see if removed
data.drop(data[data.Age > 90].index, inplace=True)
len(data.index)


# copy data so have different versions for analysis
data_kmeans = data.copy()
data_hier = data.copy()

# feature scaling

scaler = StandardScaler()
data_kmeans = pd.DataFrame(scaler.fit_transform(
    data_kmeans), columns=data_kmeans.columns)

# Calculate loss for Kmeans K values 1 to 8


loss = []
for i in range(1, 8):
    km = KMeans(n_clusters=i).fit(data_kmeans)
    loss.append(km.inertia_)

plt.plot(range(1, 8), loss)
plt.title('Finding Optimal Clusters via Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('loss')
plt.show
# carry out kmeans clustering 3 clusters
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(data_kmeans)
# see first 5 labels
kmeans.labels_[:5]
kmeans.labels_[:10]
# Add labels to data_kmenans
data_kmeans["Segment"] = kmeans.labels_


# carry out kmeans clustering with 4 clusters
kmeans2 = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans2.fit(data_kmeans)
# Add labels to data_kmenans
data_kmeans["Segment2"] = kmeans2.labels_


# Scale data_hier
data_hier = pd.DataFrame(scaler.fit_transform(
    data_hier), columns=data_hier.columns)

# training hierachical clustering model

features = data_hier.values
hc_model = AgglomerativeClustering(
    n_clusters=4,
    affinity='euclidean',
    linkage='ward'
)

hc_model.fit_predict(features)
hc_model.labels_

clusters = hierarchy.linkage(features, method="complete")
clusters


def plot_dendrogram(clusters):
    dendrogram = hierarchy.dendrogram(
        clusters, labels=hc_model.labels_, orientation="top", leaf_font_size=9, leaf_rotation=360)
    plt.ylabel('Euclidean Distance')


plot_dendrogram(clusters)


clusters = hierarchy.linkage(features, method="single")

clusters[:5]

plot_dendrogram(clusters)

clusters = hierarchy.linkage(features, method="average")
clusters[:5]
plot_dendrogram(clusters)

clusters = hierarchy.linkage(features, method="ward")
clusters[:5]
plot_dendrogram(clusters)


# Code above did not use 4 clusters due to not being changed
data["Segment"] = kmeans.labels_


# It was noticed that the number of clusters was not updated properly
# so it was decided to run this again carry out kmeans clustering with 4 clusters
kmeans3 = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans3.fit(data_kmeans)
kmeans3.labels_[:5]

# Add labels to data
data["Segment2"] = kmeans3.labels_

# Use the clustering values Segment and Segment2 to create charts showing the two clusters
# Histogam

sns.set(style="whitegrid")

sns.countplot(x='Segment', data=data)
sns.countplot(x='Segment2', data=data)

# Swarm plots showing age by clusters
ax = sns.swarmplot(x="Segment", y="Age", data=data)
plt.title('Age Segment')

ax = sns.swarmplot(x="Segment2", y="Age", data=data)
plt.title('Age Segment2')

ax = sns.swarmplot(x="Education", y="Segment", data=data)
plt.title('Education Segment')
ax = sns.swarmplot(x="Education", y="Segment2", data=data)
plt.title('Education Segment2')

ax = sns.swarmplot(x="Segment", y="Household", data=data)
plt.title('Household Segment')
ax = sns.swarmplot(x="Segment2", y="Household", data=data)
plt.title('Age Segment2')

# Stripplots

ax = sns.stripplot(x="Segment", y="Age", data=data)
plt.title('Age Segment')

ax = sns.stripplot(x="Segment2", y="Age", data=data)
plt.title('Age Segment2')

ax = sns.stripplot(x="Segment", y="Education", data=data)
plt.title('Education Segment')
ax = sns.stripplot(x="Segment2", y="Education", data=data)
plt.title('Education Segment2')

ax = sns.stripplot(x="Segment", y="Household", data=data)
plt.title('Household Segment')
ax = sns.stripplot(x="Segment2", y="Household", data=data)
plt.title('Household Segment2')

ax = sns.stripplot(x="Segment", y="Income", data=data)
plt.title('Segment Income')
ax = sns.stripplot(x="Segment2", y="Income", data=data)
plt.title('Segment2 Income')

# Kernal Density plots
sns.displot(data=data, x="Age", hue="Segment",
            multiple="stack", kind="kde", color="pastel")
plt.title('Age Distribution for Segment')
sns.displot(data=data, x="Age", hue="Segment2",
            multiple="stack", kind="kde", color="pastel")
plt.title('Age Distribution for Segment2')


sns.displot(data=data, x="Income", hue="Segment",
            multiple="stack", kind="kde", color="dark")
plt.title('Income Distribution for Segment')
sns.displot(data=data, x="Income", hue="Segment2",
            multiple="stack", kind="kde", color="dark")
plt.title('Income Distribution for Segment2')

sns.displot(data=data, x="Income", hue="Segment", col="Segment")
plt.title('Income Distribution for Segment')
sns.displot(data=data, x="Income", hue="Segment2", col="Segment2")
plt.title('Income Distribution for Segment2')

sns.displot(data=data, x="Age", hue="Segment", col="Segment")
plt.title('Age Distribution for Segment')
sns.displot(data=data, x="Age", hue="Segment2", col="Segment2")
plt.title('Age Distribution for Segment2')

sns.displot(data=data, x="Recency", hue="Segment", col="Segment")
sns.displot(data=data, x="Recency", hue="Segment2", col="Segment2")

sns.displot(data=data, x="TotalCmp", hue="Segment", col="Segment")
sns.displot(data=data, x="TotalCmp", hue="Segment2", col="Segment2")

# Scatterplot
sns.set_palette("bright")
sns.scatterplot(x="Income", y="Recency", data=data, hue="Segment")
sns.scatterplot(x="Income", y="Recency", data=data, hue="Segment2")

sns.scatterplot(x="Income", y="TotalCmp", data=data, hue="Segment")
sns.scatterplot(x="Income", y="TotalCmp", data=data, hue="Segment2")

sns.pairplot(data = data,
             x_vars=["Income"],
             y_vars=["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"],
             hue="Segment")


sns.pairplot(data = data,
             x_vars=["Income"],
             y_vars=["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"],
             hue="Segment2")


sns.pairplot(data = data,
             x_vars=["Income", "TotalCmp", "Response"],
             y_vars=["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth"],
             hue="Segment")


sns.pairplot(data = data,
             x_vars=["Income", "TotalCmp", "Response"],
             y_vars=["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth"],
             hue="Segment2")



