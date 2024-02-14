# ML-Project-IMDB
# Sentiment Analysis using Support Vector Machines

This Python script performs sentiment analysis on a given dataset using Support Vector Machines (SVMs). Sentiment analysis involves determining the sentiment or opinion expressed in a piece of text, typically categorized as positive or negative.

# 1.	Dependencies
•	nltk: Natural Language Toolkit for text processing tasks.
•	pandas: Library for data manipulation and analysis.
•	re: Regular expression operations for string manipulation.
•	BeautifulSoup: Library for pulling data out of HTML and XML files.
•	sklearn: Library providing tools for machine learning tasks.
•	WordNetLemmatizer: Lemmatization tool from NLTK for word normalization.
•	stopwords: NLTK corpus containing common stop words.
# 2. Usage
•	Ensure you have all dependencies installed. You can install them via pip: pip install nltk pandas scikit-learn beautifulsoup4.
•	Make sure you have a dataset file named Dummy Dataset.csv in the same directory as the script.
•	Run the script.
# 3.	Description
•	The script imports necessary libraries including NLTK, Pandas, BeautifulSoup, and scikit-learn modules.
•	It defines functions for text preprocessing including HTML tag removal, text cleaning, and lemmatization.
•	The dataset is loaded using Pandas, where sentiment labels are converted to binary values (0 for negative and 1 for positive).
•	Text preprocessing is applied to the reviews column in the dataset to clean and normalize the text.
•	The dataset is split into training and testing sets.
# 4.	Bag of Words (BoW) vectorization is performed on the processed text data.
•	An SVM model is trained on the BoW vectors.
•	Classification report is printed to evaluate the SVM model's performance.
•	A pipeline is created for performing grid search to find the optimal hyperparameters for the SVM model.
•	Grid search is conducted using cross-validation to find the best combination of hyperparameters.
•	Results of the grid search, including the best score and parameters, are printed.
# 5.	Note
•	Ensure that the dataset file IMDB Dataset.csv is present in the same directory as the script.
•	Depending on the size of your dataset and computational resources, grid search may take a considerable amount of time to complete.
# •	Author
This script is authored by Muhammad Talha Riaz.
