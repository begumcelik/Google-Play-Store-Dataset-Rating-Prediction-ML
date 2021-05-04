# Google Play Store Dataset Rating Prediction

The main dataset, which contains information about Google Play Store apps on various categories, will be investigated from data scientist’ point of view to create the down-to-earth statistics that is related to usage of applications in many cases. As an example, one of the areas of utilization of this interpretation, can be more reliable Google Play Store market strategies that can be derived from those statistics.

The steps that will be taken to reach the goal of creating realist and useful statistics, are listed as:

1. Formulating the questions that will be asked to bring into the open the relations that have a significant meaning for that dataset
2. Cleaning the data which has provided, not collected by the contributors of the project
3. Data analysis should be done after the manipulation and clearance of the data
4. Patterns in data can be identified by using visualization
5. After that process, the conclusions should be interpreted to reveal the answers to the questions that has been asked

# Google Play Store Apps Dataset

The dataset consists of two sperate data source files. The main file which is named as googleplaystore includes 13 columns that each one gives different types of information which is scraped from the Google Play Store. The column names and the information stored in those columns are explained below.

1. App: Application Name
2. Category: Classification of the apps according to subject
3. Rating: Overall user rating of the app
4. Reviews: Number of user reviews for the
5. Size: Size of the app in terms of mb,gb
6. Installs: Number of user downloads/installs for the app
7. Type: Free or paid info
8. Price: Price of the apps in terms of dolar
9. Content Rating: Age group that app appeals to(Children / Mature 21+ / Adult)
10. Genres: An app can belong to multiple genres (apart from its main category)
11. Last Updated: The last update date of the app
12. Current Ver: Current version number of the app
13. Android Ver: The minimum android version required to be able to use the app

The second source only contains information about the 100 reviews for the each individual app. Columns of the googleplaystore_user_reviews file are listed below.

1. App: Application Name
2. Translated_Review:User comments (Preprocessed and translated to English)
3. Sentiment: Type of the comment (Positive/Negative/Neutral) (Preprocessed)
4. Sentiment_Polarity: Sentiment polarity score
5. Sentiment_Subjectivity: Sentiment subjectivity score

Dataset can be found at https://www.kaggle.com/lava18/google-play-store-apps

# Project Summary

The problem is "Which attributes are having importance on the application's rating?" Our objective is finding the relationship between rating and the other attributes in the dataset Google Play Store Dataset.

# Problem Description
The first question that will be searched for is about the existence of correlation between pricing and rating of the google play store apps. If exists, is it possible to claim that the high price apps are more qualified or not?

Another question that can be asked is which categories and genres appeal to which age group ,that is named as content rating in the dataset.

Additionally, the correlation between number of reviews and rating can be searched and the result can give information about the content of the reviews. Are the reviews mostly negative or positive?

Can the Rating be predicted precisely, using other attributes of the dataset?

# Methods

1. Cleaning the data
2. Data manipulation
3. Data Visualization
4. Machine Learning

# Steps Taken

1. Determine null values and convert to NaN
2. Fill Nan values with mean of the column
3. Replace characters with more useful ones
4. Convert data types of the columns
5. Determine intervals for large values to simplify the data
6. Scatter plot graph
7. Histogram plot
8. KNN method to predict Rating
9. Linear Regression method to predict Rating
10. RF method to predict Rating

# Results

Correlation between Rating and Pricing is 0.01854348528168532 -nearly 0-.

Results after the Linear Regression method:

Actual mean of ratings: 3.523528830112766
Predicted mean: 3.5358888375748214
Accuracy: 91.46322764129208 %
Standard Deviation of actual: 1.5855879954364303
Standard Deviation of predicted: 1.5072537346657886
Accuracy of knn is : 91.49%

Accuracy of rf is : 0.6458910433979687

Mean Absolute Error: 0.43351800554016623
Mean Squared Error: 0.5923361034164358
Root Mean Squared Error: 0.7696337462822403

# Conclusion

1. Rating and pricing attributes are not corrolated.
2. Last Updated attribute affects the Rating of the application the most.
3. Rating can be predicted with 90% accuracy without using all attributes of the dataset.
4. Linear Regression Method and KNN Method give more accurate prediction of Rating than Random Forest Method.

In conclusion, we saw that it is possible to predict a value of an attribute with high accuracy. Using Machine Learning is useful for future estimations and also an important way to avoid future mistakes

# Contributers

Ardacan Dağ

Begüm Çelik

Gülçin Uras

Güren İçim

İbrahim Murat Karakaya
