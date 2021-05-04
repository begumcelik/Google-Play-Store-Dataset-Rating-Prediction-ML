# -*- coding: utf-8 -*-

The main dataset, which contains information about Google Play Store apps on various categories, will be investigated from data scientist’ point of view 
to create the down-to-earth statistics that is related to usage of applications in many cases. As an example, one of the areas of utilization of this interpretation,
can be more reliable Google Play Store market strategies that can be derived from those statistics. 

The steps that will be taken to reach the goal of creating realist and useful statistics, are listed as:

*   Formulating the questions that will be asked to bring into the open the relations that have a significant meaning for that dataset
*   Cleaning the data which has provided, not collected by the contributors of the project
*   Data analysis should be done after the manipulation and clearance of the data
*   Patterns in data can be identified by using visualization
*   After that process, the conclusions should be interpreted to reveal the answers to the questions that has been asked

# **Google Play Store Apps Dataset**

The dataset consists of two sperate data source files. The main file which is named as googleplaystore includes 13 columns that each one gives different types of information which is scraped from the Google Play Store. The column names and the information stored in those columns are explained below.


1.   App: Application Name
2.   Category: Classification of the apps according to subject
3.   Rating: Overall user rating of the app
4.   Reviews: Number of user reviews for the 
5.   Size: Size of the app in terms of mb,gb
6.   Installs: Number of user downloads/installs for the app
7.   Type: Free or paid info
8.   Price: Price of the apps in terms of dolar
9.   Content Rating: Age group that app appeals to(Children / Mature 21+ / Adult)
10.  Genres: An app can belong to multiple genres (apart from its main category)
11.  Last Updated: The last update date of the app
12.  Current Ver: Current version number of the app
13.  Android Ver: The minimum android version required to be able to use the app

The second source only contains information about the 100 reviews for the each individual app. Columns of the googleplaystore_user_reviews file are listed below.

1.  App: Application Name
2.  Translated_Review:User comments (Preprocessed and translated to English)
3.  Sentiment: Type of the comment (Positive/Negative/Neutral) (Preprocessed)
4.  Sentiment_Polarity: Sentiment polarity score
5.  Sentiment_Subjectivity: Sentiment subjectivity score



[dataset](https://www.kaggle.com/lava18/google-play-store-apps)

# **INTRODUCTION**

>**Summary**

>The problem is "Which attributes are having importance on the application's rating?"
Our objective is finding the relationship between rating and the other attributes in the dataset Google Play Store Dataset. 

 >**Methods**
 

1.   Cleaning the data
2.   Data manipulation
3.   Data Visualization
4.   Machine Learning   

 >**Results**

 >Rating and pricing attributes are not corrolated.

 >Last Updated attribute affects the Rating of the application the most.

 >Rating can be predicted with 90% accuracy without using all attributes of the dataset.

 **Problem description**

The first question that will be searched for is about the existence of correlation between pricing and rating of the google play store apps. If exists, is it possible to claim that the high price apps are more qualified or not?

 >Another question that can be asked is which categories and genres appeal to which age group ,that is named as content rating in the dataset. 

 >Additionally, the correlation between number of reviews and rating can be searched and the result can give information about the content of the reviews. Are the reviews mostly negative or positive?

 >Can the Rating be predicted precisely, using other attributes of the dataset?

 **Methods:**



*   Determine null values and convert to NaN
*   Fill Nan values with mean of the column
*   Replace characters with more useful ones
*   Convert data types of the columns
*   Determine intervals for large values to simplify the data
*   Scatter plot graph
*   Histogram plot
*   KNN method to predict Rating
*   Linear Regression method to predict Rating
*   RF method to predict Rating

 **Result:**

>Correlation between Rating and Pricing is 0.01854348528168532 -nearly 0-.

>**Results after the Linear Regression method:**

*   Actual mean of ratings:  3.523528830112766
*   Predicted mean:  3.5358888375748214
*   Accuracy:  91.46322764129208 %
*   Standard Deviation of actual:  1.5855879954364303
*   Standard Deviation of predicted:  1.5072537346657886

>**Accuracy of knn is : 91.49%**

>**Accuracy of rf is : 0.6458910433979687**

*   Mean Absolute Error:  0.43351800554016623
*   Mean Squared Error:  0.5923361034164358
*   Root Mean Squared Error:  0.7696337462822403

**Discussion:**

*   We can conclude that Pricing and Rating are not correlated.
*   Linear Regression Method and KNN Method give more accurate prediction of Rating than Random Forest Method.


**Conclusion:**
In conclusion, we saw that it is possible to predict a value of an attribute with high accuracy. Using Machine Learning is useful for future estimations and also an important way to avoid future mistakes


"""

# Commented out IPython magic to ensure Python compatibility.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
# figure size in inches

# %matplotlib inline
from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive/googleplaystore.csv"
path2 = "/content/drive/My Drive/googleplaystore_user_reviews.csv"
data = pd.read_csv(path)
data_reviews = pd.read_csv(path2) #Reviews' csv file.
data.head(5)

data_reviews.head(5)  #data_reviews has read proof.

from google.colab import drive
drive.mount('/content/drive')

print(data.shape) #shape of Play Store Data.

print(data_reviews.shape) #shape of Data reviews.

"""# **Cleaning the Data**"""

#Replacing all the unknown values in our DataFrame to NaN
def replaceWithNaN(df):
    '''
    df: (dataframe) input dataframe 
    returns: (dataframe) modified dataframe 
    '''
    df= df.replace('?',np.NaN)
    df= df.replace('n.a',np.NaN)
    df= df.replace(' ',np.NaN)
    return df

data=replaceWithNaN(data)
data_reviews=replaceWithNaN(data_reviews)

#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)

#missing data
total = data_reviews.isnull().sum().sort_values(ascending=False)
percent = (data_reviews.isnull().sum()/data_reviews.isnull().count()).sort_values(ascending=False)
missing_data2 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data2.head(6)

#dropped NaN values from data_reviews 
data_reviews.dropna(inplace=True)

data.fillna(value = data.mean(), inplace=True)
#missing data
data.dropna(inplace=True)

print(data.shape)

print(data_reviews.shape)

data['Rating'].describe()

# information about data
data.info()

"""# **Data Manipulation**

**We need to convert columns including string to numbers. In order to do that preprocessing have used. In order to be able to make better predictions, we should include string based columns encoded.**
"""

#Cleaning 'Price' column with removing '$' signs and converting string values to numerical values
data['Price']=data['Price'].str.replace('$','')
data[[ "Price"]] = data[["Price"]].apply(pd.to_numeric)
data.sort_values('Price')

from sklearn import preprocessing
# App 
encoder = preprocessing.LabelEncoder()
data['App'] = encoder.fit_transform(data['App'])

#Genres
encoder = preprocessing.LabelEncoder()
data['Genres'] = encoder.fit_transform(data['Genres'])

#Content Rating 
encoder = preprocessing.LabelEncoder()
data['Content Rating'] = encoder.fit_transform(data['Content Rating'])

#Converting 'Reviews' columns values to integer
data['Reviews'] = data['Reviews'].astype(int)

#Converting 'Price' columns values to integer
data['Price'] = data['Price'].astype(int)

#Removing ',' and '+' signs from 'Installs' column
data['Installs']=data['Installs'].str.replace(',','')
data['Installs']=data['Installs'].str.replace('+','')

# Last Updated encoding
import datetime
import time 
data['Last Updated'] = data['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))

data['Current Ver']=data['Current Ver'].str.replace('.','')
data['Current Ver']=data['Current Ver'].str.replace('Varies with device','0000')
data['Current Ver']=data['Current Ver'].str.replace('Build','0000')
data['Current Ver']=data['Current Ver'].str.replace(' ','')
#data['Current Ver'] = pd.to_numeric(data['Current Ver'])
encoder = preprocessing.LabelEncoder()
data['Current Ver'] = encoder.fit_transform(data['Current Ver'])

# Convert kbytes to Mbytes 
k_indices = data['Size'].loc[data['Size'].str.contains('k')].index.tolist()
converter = pd.DataFrame(data.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))
data.loc[k_indices,'Size'] = converter
# Size cleaning
data['Size'] = data['Size'].apply(lambda x: x.strip('M'))
data[data['Size'] == 'Varies with device'] = 0
data['Size'] = data['Size'].astype(float)

"""# *Is there any correlation between pricing and rating of the Google Play Store apps? Can one claim that as apps are getting pricier the quality of the application increases?*"""

# Visualization for Rating and Price
plt.figure(figsize=(10,20))
plt.scatter(data["Rating"], data["Price"])
#.sort_values(ascending=False) can be added

#correlation between Price and Rating
data['Rating'].corr(data['Price'])
#Ratings and prices are not correlated

#Grouped data as range 50 and get the means in every 50 $ range
data.groupby(pd.cut(data["Price"], np.arange(0, 400+50,50)))["Rating"].mean()

"""**The correlation value is negative but very close to zero. But when the scatter plot is investigated, one can observe that the amount of the data that comes from the cheaper and pricier apps are not equal. In order to solve this ambiguity, price of the apps has grouped and the means of ratings has been listed. In addition to those methods the type info can be used to seperate free or paid apps' data in next step.**

# *Which categories and genres appeal to which age group ,that is named as content rating in the dataset?*
"""

#Unique Categories list
data["Category"].unique()

#Unique Content Ratings list
data["Content Rating"].unique()

# Grouped Category and Content Rating column, to see their statistics.
group = data.groupby(['Category','Content Rating'])
#group.first()
group.describe()

#After grouping these 2 columns, we found maximum values occurred in grouping those values.

#group.describe()['mean'].max()
bla = data.groupby(['Category','Content Rating'], sort=False)['Rating'].max()
bla

# Means of these values, from 2 columns 
lba = data.groupby(['Category','Content Rating'], sort=False)['Rating'].mean()
lba

# Visualization of above.
plt.hist(lba, bins=50)

"""# *Is there any relation between the type of reviews which is named as 'Sentiment' in reviews' csv and rating?*"""

#Numerated review.csv file able to convert strings to numeric.
data_reviews['Sentiment']=data_reviews['Sentiment'].str.replace('Positive','1')
data_reviews['Sentiment']=data_reviews['Sentiment'].str.replace('Negative','0')
data_reviews['Sentiment']=data_reviews['Sentiment'].str.replace('Neutral','0.5')

data_reviews[[ "Sentiment"]] = data_reviews[["Sentiment"]].apply(pd.to_numeric)

frames=[data,data_reviews]
d= pd.concat(frames, axis=1, join='inner')

new_D = d.groupby(["Rating"])["Sentiment"].mean()

new_D

d['Sentiment'].corr(d['Rating'])

"""# **Machine Learning**"""

#other way to find data_x value next line 
#data_x = data.drop(labels = ['App','Size','Category','Rating','Genres','Last Updated','Current Ver','Android Ver','Type'],axis = 1)

#Assigned X and Y value for sklearn methods 

data_x = data[['App','Genres','Installs','Reviews','Price','Content Rating','Current Ver','Size','Last Updated']]

data_y = data['Rating']

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#Split data into Train , Test 
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

#Creating Linear Regression
linear_reg = LinearRegression()

#Fitting
linear_reg.fit(x_train,y_train)

#Prediction
linear_reg_prediction = linear_reg.predict(x_test)


#Plotting our predictions and the original results
plt.figure(figsize=(20,10))
sns.regplot(linear_reg_prediction,y_test,color='blue', label = 'Marks', marker = 'x')

plt.legend()
plt.title('Linear Regression')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()

# Results after the Linear Regression method.

print("Actual mean of ratings: ",data_y.mean())
print("Predicted mean: ",linear_reg_prediction.mean())
linear_accuracy = linear_reg.score(x_test,y_test)
print("Accuracy: ", linear_accuracy*100,'%')
print("Standard Deviation of actual: ", data_y.std())
print("Standard Deviation of predicted: ", linear_reg_prediction.std())

d_y=pd.qcut(data['Rating'], q=3, precision=0)

"""data['Last Updated'] = data['Last Updated'].astype(str)
data['Last Updated'] =data['Last Updated'].apply(
    lambda row: [val for val in row if val != "0"])
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

**When the results of Linear Regression method as an machine learning tool have 
been analyzed, the scantness of the accuracy value can be observed. In order to solve this problem, more columns may be cleaned to be used as Linear Regression parameters as a next step of this project process.**
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


knn_x = data[['App','Genres','Installs','Reviews','Price','Content Rating','Current Ver','Size','Last Updated']]
knn_y = data['Rating']


X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(knn_x, knn_y, test_size=0.20,random_state=10)

knn = KNeighborsRegressor(n_neighbors=15)

knn.fit(X_knn_train, y_knn_train)

#y_pred = knn.predict(X_knn_test)

accuracy_knn = knn.score(X_knn_test,y_knn_test)
print('Accuracy of knn is : ' + str(np.round(accuracy_knn*100, 2)) + '%')

# different number of n_estimators 
n_neig = np.arange(1, 10, 1)
n_scores = []
for i in n_neig:
    knn.set_params(n_neighbors=i)
    knn.fit(X_knn_train, y_knn_train)
    n_scores.append(knn.score(X_knn_test, y_knn_test))



# Plotting the result
plt.figure(figsize=(10, 8))
plt.title("With different Estimators")
plt.xlabel("Number of Neighbors K")
plt.ylabel("Score")
plt.plot(n_neig, n_scores,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12)

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Data splitting again same as Linear Regression 
rf_x = data[['App','Genres','Installs','Reviews','Price','Content Rating','Current Ver','Size','Last Updated']]

rf_y = d_y
#preprocessing.LabelEncoder() - convert string or float values to 0 .. n classes.
#If we put as input rf_x, rf_y to fit method it cause error. To avoid it we will convert and encode labels.
lab_enc = preprocessing.LabelEncoder()
new_rf_y = lab_enc.fit_transform(rf_y)

#Creating Random Forest Classifier 
rf = RandomForestClassifier(n_estimators=100,random_state=1)

#Train test split
rf_x_train, rf_x_test, rf_y_train, rf_y_test = train_test_split(rf_x, new_rf_y, test_size=0.2)

#fit the train values
rf.fit(rf_x_train,rf_y_train)
#Prediction with Random Forest Classifier
rf_prediction = rf.predict(rf_x_test)
#Accuracy score of our predictions .
accuracy_score_rf = accuracy_score(rf_prediction,rf_y_test)
print(accuracy_score_rf)

from sklearn.metrics import confusion_matrix

#Visualization: 
plt.figure(figsize=(20,10))
#sns.regplot(rf_prediction,rf_y_test,color='blue', label = 'Integer', marker = 'x')
matrix = confusion_matrix(rf_y_test,rf_prediction)

plt.figure(figsize=(20,10))
plt.title("Confusion Matrix our Prediction and Original Values")
sns.heatmap(matrix,annot=True,cmap="Greens",fmt="d",cbar=True)

#Accuracy of our predictions 
print("Accuracy: ",accuracy_score_rf)

different_rf = RandomForestRegressor(n_jobs=-1)
# Try different numbers of n_estimators - this will take a minute or so
n_estimators = np.arange(10, 100, 10)
new_rf_scores = []
for i in n_estimators:
    different_rf.set_params(n_estimators=i)
    different_rf.fit(rf_x_train, rf_y_train)
    new_rf_scores.append(different_rf.score(rf_x_test,rf_y_test))

plt.figure(figsize=(10, 8))
plt.title("With using Estimators")
plt.xlabel("# of estimator")
plt.ylabel("Scores")
plt.plot(n_estimators, new_rf_scores,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12)

from sklearn import metrics
rf_predictions_est = rf.predict(rf_x_test)
abs_mean_err = metrics.mean_absolute_error(rf_y_test,rf_predictions_est)
print("Mean Absolute Error: ",abs_mean_err )

mean_sq_err = metrics.mean_squared_error(rf_y_test,rf_predictions_est)
print("Mean Squared Error: ", mean_sq_err)

r_mean_sq_err = np.sqrt(metrics.mean_squared_error(rf_y_test,rf_predictions_est))
print("Root Mean Squared Error: ", r_mean_sq_err)

"""**This prediction accuracy which has been achieved by Random Forest Classifier method, is higher than the previous one.
However, in order to achieve more reliable results in the next step of this project, more column should be cleaned and taken into account to increase prediction accuracy.**"""

# This part is for comparing our Prediction Methods 

# Linear Regression accuray is lower when it is compared with the Random Forest Classifier.
acc = pd.Series(data=[linear_accuracy,accuracy_score_rf,accuracy_knn],index=['Linear Regression','RandomForestClassifier','KNN'])
fig= plt.figure(figsize=(10,10))
acc.sort_values().plot.bar()
plt.title('Accuracy of all models')


