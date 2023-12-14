#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[2]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[3]:


from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold


# In[4]:


import os
dir_path = '/Users/sophie/Desktop'

# List files in the directory
files = os.listdir(dir_path)

# Assuming your data file is a CSV file named 'IMDB Dataset.csv'
data = pd.read_csv(os.path.join(dir_path, 'IMDB_Dataset.csv'))


# In[5]:


# Descriptive statistics (class distribution)
class_distribution = data['sentiment'].value_counts()
print("\nClass Distribution:\n", class_distribution)

plt.figure(figsize=(4, 4))
class_distribution.plot.pie(autopct='%1.1f%%', startangle=90, colors= ['skyblue', 'lightcoral'])
plt.title('Class Distribution')
plt.ylabel('')
plt.axis('equal')
plt.show()


# # Data Preprocessing

# In[6]:


def preprocess(text):
    # Clean text
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords, personal pronouns, determiners, coordinating conjunctions, and prepositions
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words and word.lower() not in ["he", "she", "i", "we", "the", "a", "an", "another", "for", "and", "nor", "but", "or", "yet", "so", "in", "under", "towards", "before"]]

    # From adv to adj
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='a') for word in words]

    # Join words back into sentence
    processed_text = ' '.join(words)

    return processed_text


# In[7]:


data['processed_reviews'] = data['review'].apply(preprocess)
data.head()


# ## Word Cloud

# In[8]:


# Function to extract adj from text
def extract_adjectives(text):
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)
    adjectives = [word.lower() for word, tag in tagged_words if tag.startswith('JJ')]
    return ' '.join(adjectives)

data['adjective_reviews'] = data['processed_reviews'].apply(extract_adjectives)

# Combine all adj into a string
adjective_text = ' '.join(data['adjective_reviews'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(adjective_text)


# In[9]:


positive_reviews = data[data['sentiment'] == 'positive']['adjective_reviews']
negative_reviews = data[data['sentiment'] == 'negative']['adjective_reviews']

# Combine all adj for each sentiment
positive_adjective_text = ' '.join(positive_reviews)
negative_adjective_text = ' '.join(negative_reviews)

# Word Cloud for positive adj
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_adjective_text)

# Word Cloud for negative adj
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_adjective_text)

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Adjective Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Adjective Word Cloud')
plt.axis('off')

plt.show()


# # Sentiment Analysis

# In[10]:


with open('/Users/sophie/Desktop/positive_words.txt', 'r') as file:
    positive_text = file.read()
with open('/Users/sophie/Desktop/negative_words.txt', 'r') as file:
    negative_text = file.read()

sentiment_dict = list(set(word_tokenize(positive_text) + word_tokenize(negative_text)))
print(len(sentiment_dict))


# In[11]:


def filter_words(word_list, training_data):
    # Tokenize training data & create unique words set
    unique_words_set = set(word for text in training_data for word in word_tokenize(text))

    # Filter list based on the unique words
    filtered_words = [word for word in word_list if word in unique_words_set]

    return filtered_words

filtered_word_list = filter_words(sentiment_dict, data['processed_reviews'])
print(len(filtered_word_list))


# In[12]:


def encode_and_select_features(text_vector, word_list, target_variable, num_features_to_select):
    
    tokenized_text = [word_tokenize(text) for text in text_vector]
    tfidf_vectorizer = TfidfVectorizer(vocabulary=word_list)
    tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(tokens) for tokens in tokenized_text])

    # Selecting K best features using chi-squared test
    selector = SelectKBest(score_func=chi2, k=num_features_to_select)
    selected_features = selector.fit_transform(tfidf_matrix, target_variable)

    return selected_features

num_features_to_select = 300
selected_features = encode_and_select_features(data['processed_reviews'], filtered_word_list, data['sentiment'],
                                               num_features_to_select)
reduced_matrix= selected_features.toarray()


# In[13]:


print(reduced_matrix.shape)
print(reduced_matrix)


# # Model Training

# In[14]:


X = reduced_matrix
y = data['sentiment'].map({'positive': 1, 'negative': 0}).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# In[15]:


# Function for plotting ROC curves
def plot_roc_curve(fpr, tpr, model_name, auc_score):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (auc = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='best')
    plt.show()


# In[16]:


# Logistic Regression
model = LogisticRegression(max_iter=1000)
name = 'Logistic Regression'
model.fit(X_train, y_train)

predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, digits=4)
print(f"{name} Accuracy: {test_accuracy}")
print(report)

# ROC Curve
proba_predictions = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_predictions)
auc_score = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, name, auc_score)


# In[17]:


# KNN
name = 'KNN'
knn = KNeighborsClassifier()

param_grid = {'n_neighbors': [1, 5, 13, 17, 25, 77,93]}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
test_accuracy = grid_search.score(X_test, y_test)
predictions = grid_search.predict(X_test)
report = classification_report(y_test, predictions, digits=4)
print("Best Parameters: ", grid_search.best_params_)
print(f"{name} Accuracy: {test_accuracy}")
print("Classification Report:\n", report)

# KNN ROC Curve
proba_predictions = grid_search.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_predictions)
auc_score = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, name, auc_score)


# In[18]:


# Decision Tree
name = "Decision Tree"
dt_classifier = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 4, 8],
    'min_samples_split': [1, 2, 5],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
test_accuracy = grid_search.score(X_test, y_test)
predictions = grid_search.predict(X_test)
report = classification_report(y_test, predictions, digits=4)
print("Best Parameters: ", grid_search.best_params_)
print(f"{name} Accuracy: {test_accuracy}")
print("Classification Report:\n", report)

# DT ROC Curve
proba_predictions = grid_search.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_predictions)
auc_score = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, name, auc_score)


# In[19]:


# Random Forest
name = 'Random Forest'
rf_classifier = RandomForestClassifier()

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 30],
    'min_samples_split': [1, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
test_accuracy = grid_search.score(X_test, y_test)
predictions = grid_search.predict(X_test)
report = classification_report(y_test, predictions, digits=4)
print("Best Parameters: ", grid_search.best_params_)
print(f"{name} Accuracy: {test_accuracy}")
print("Classification Report:\n", report)

# RF ROC Curve
proba_predictions = grid_search.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_predictions)
auc_score = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, name, auc_score)


# In[20]:


# LDA: check the assumption on same covariance
classes = np.unique(y)
cov_matrices = []

# Calculate the covariance matrix for each class
for class_val in classes:
    class_data = X[y == class_val]

    if class_data.ndim == 2:
        cov_matrix = np.cov(class_data.T)
        cov_matrices.append(cov_matrix)
        print(f"Covariance Matrix for Class {class_val}:\n{cov_matrix}\n")

# Compare the covariance matrices
n_classes = len(cov_matrices)
for i in range(n_classes):
    for j in range(i+1, n_classes):
        diff = cov_matrices[i] - cov_matrices[j]
        frob_norm = np.linalg.norm(diff, 'fro')  # Frobenius norm
        print(f"Difference between covariance matrices of class {i} and class {j}: {frob_norm}")


# In[21]:


# LDA
model = LinearDiscriminantAnalysis()
name = 'LDA'
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, digits=4)
print(f"{name} Accuracy: {accuracy}")
print(report)

# ROC Curve
proba_predictions = model.predict_proba(X_test)[:, 1]  # Probabilities needed for ROC curve
fpr, tpr, _ = roc_curve(y_test, proba_predictions)
auc_score = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, name, auc_score)


# In[22]:


# QDA
model = QuadraticDiscriminantAnalysis()
name = 'QDA'
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, digits=4)
print(f"{name} Accuracy: {accuracy}")
print(report)

# ROC Curve
proba_predictions = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_predictions)
auc_score = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, name, auc_score)


# In[ ]:




