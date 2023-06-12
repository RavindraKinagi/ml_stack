sentiment analysis using the Twitter Tweets Sentiment Dataset. The code performs the following steps:
1.	Imports the necessary libraries, including numpy and pandas.
2.	Reads the dataset from the CSV file.
3.	Checks for null values in the dataset.
4.	Removes rows with null values.
5.	Resets the index of the dataframe.
6.	Splits the dataset into training and testing data.
7.	Uses CountVectorizer and Multinomial Naive Bayes to create a pipeline for the sentiment analysis model.
8.	Fits the model on the training data.
9.	Calculates the accuracy of the model on the testing data.
10.	Makes predictions on the testing data.
11.	Calculates various evaluation metrics such as accuracy, precision, recall, and F1-score.


