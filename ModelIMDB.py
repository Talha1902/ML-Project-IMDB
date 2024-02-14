import nltk
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Import dataset and replace labels with 0 and 1 for classification
df = pd.read_csv('Dummy Dataset.csv', encoding='Latin-1')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Define stop_words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Define strip_html function to handle text and file input
def strip_html(text):
    if isinstance(text, str):
        return BeautifulSoup(text, "html.parser").get_text()
    else:
        # If it's not a string, assume it's a file name and read its contents
        with open(text, 'r', encoding='utf-8') as file:
            return BeautifulSoup(file.read(), "html.parser").get_text()

# Define clean_text function
def clean_text(text):
    text = strip_html(text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

# Create a new column for processed reviews
df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))
# Display all the data after cleaning
pd.set_option('display.max_columns', None)  # Show all columns
# print(df)
df.head(20)

#Defining input and target variable
x = df['Processed_Reviews']
y = df['sentiment']

#Training and splitting
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Vectorization and Bag of words method with default parameters
count_vect = CountVectorizer().fit(df['Processed_Reviews'].values.astype('U'))
bow_train = count_vect.transform(X_train.values.astype('U'))
bow_test = count_vect.transform(X_test.values.astype('U'))

#instantiate the model (using the default parameters)
SVM = SVC()

# fit the model with pre-processed data
SVM.fit(bow_train, y_train)

#perform classification and prediction on samples in tf_test
predicted_SVM = SVM.predict(bow_test)
print(classification_report(y_test, predicted_SVM))

#Creating a Pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('SVM', SVC())
])

#Defining hyperparameters
parameters = {
    'vect__max_df':[0.1,0.2,0.3,0.4,0.5,0.6,0.7],
    'vect__ngram_range':  [(1,1), (1,2), (1,3)],
    'SVM__kernel': ['poly', 'rbf', 'sigmoid'],
    'SVM__C': [50, 10, 1.0, 0.1, 0.01]}

# define grid search
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(pipeline, param_grid=parameters, refit = True, verbose = 3, cv=5)
grid_result = grid_search.fit(df.loc[:100, 'Processed_Reviews'].values.astype('U'), df.loc[:100, 'sentiment'].values.astype('U'))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))