# ML-Project-IMDB
# Sentiment Analysis using Support Vector Machines

This Python script performs sentiment analysis on a given dataset using Support Vector Machines (SVMs). Sentiment analysis involves determining the sentiment or opinion expressed in a piece of text, typically categorized as positive or negative.

# 1.	Dependencies
1.	nltk: Natural Language Toolkit for text processing tasks.
2.	pandas: Library for data manipulation and analysis.
3.	re: Regular expression operations for string manipulation.
4.	BeautifulSoup: Library for pulling data out of HTML and XML files.
5.	sklearn: Library providing tools for machine learning tasks.
6.	WordNetLemmatizer: Lemmatization tool from NLTK for word normalization.
7.	stopwords: NLTK corpus containing common stop words.

# 2. Usage
1.	Ensure you have all dependencies installed. You can install them via pip: pip install nltk pandas scikit-learn beautifulsoup4.
2.	Make sure you have a dataset file named Dummy Dataset.csv in the same directory as the script.
3.	Run the script.

# 3.	Description
1.	The script imports necessary libraries including NLTK, Pandas, BeautifulSoup, and scikit-learn modules.
2.	It defines functions for text preprocessing including HTML tag removal, text cleaning, and lemmatization.
3.	The dataset is loaded using Pandas, where sentiment labels are converted to binary values (0 for negative and 1 for positive).
4.	Text preprocessing is applied to the reviews column in the dataset to clean and normalize the text.
5.	The dataset is split into training and testing sets.

# 4.	Bag of Words (BoW) vectorization is performed on the processed text data.
1.	An SVM model is trained on the BoW vectors.
2.	Classification report is printed to evaluate the SVM model's performance.
3.	A pipeline is created for performing grid search to find the optimal hyperparameters for the SVM model.
4.	Grid search is conducted using cross-validation to find the best combination of hyperparameters.
5.	Results of the grid search, including the best score and parameters, are printed.

# 5.	Note
1.	Ensure that the dataset file IMDB Dataset.csv is present in the same directory as the script.
2.	Depending on the size of your dataset and computational resources, grid search may take a considerable amount of time to complete.

# •	Author
This script is authored by Muhammad Talha Riaz.
