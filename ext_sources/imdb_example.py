#%%

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import np
import matplotlib as plt

#%%

# Load training data with training texts and training labels
# This is an sklearn type "bunch"
reviews_train = load_files("./ext_sources/data/aclImdb/train/")
text_train, y_train = reviews_train.data, reviews_train.target
print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[6]: \n{}".format(text_train[6]))

#%%

# We have lots of html specific line breaks in the documents
# They need to be removed
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

#%%

# How many positive/ negative tagged documents are there
print("Samples per class (training): {}".format(np.bincount(y_train)))

#%%

reviews_test = load_files("./ext_sources/data/aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
print("Number of documents in test data: {}".format(len(text_test)))
print("Samples per class (test): {}".format(np.bincount(y_test)))
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

# TODO: Assign label negative or positive based on the text content

#%%

#-----------------------------------------------------------------------------------------------------------------------
# A simple test for vectorization process
bards_words = ["The fool doth think he is wise,",
               "but the wise man knows himself to be a fool"]
# Create vocabulary
vect = CountVectorizer()
vect.fit(bards_words)
print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("Vocabulary content:\n {}".format(vect.vocabulary_))
# Transform it
bag_of_words = vect.transform(bards_words)
print("bag_of_words: {}".format(repr(bag_of_words)))
print("Dense representation of bag_of_words:\n{}".format(bag_of_words.toarray()))
#-----------------------------------------------------------------------------------------------------------------------

#%%

vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print("X_train:\n{}".format(repr(X_train)))
# The vocabulary contains 74849 elements, thus the number of rows = 74849
# Columns represent the documents

#%%

# Another way to access vocabulary
feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))
print("First 20 features:\n{}".format(feature_names[:20]))
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 2000th feature:\n{}".format(feature_names[::2000]))

#%%

# YOUAREHERE

# TODO: test logistic regression further
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)

#%%

X_test = vect.transform(text_test)
print("Test score: {:.2f}".format(grid.score(X_test, y_test)))

#%%

## TODO: TRY THIS 1
## TODO solved: Gensim equivalent is applied
# We only want to return the words that appear at least in 5 different documents
# which is to define with min_df argument
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("X_train with min_df=5: {}".format(repr(X_train)))
# Now we have 27271 elements in vocabulary

#%%

feature_names = vect.get_feature_names()
print("First 50 features:\n{}".format(feature_names[:50]))
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 700th feature:\n{}".format(feature_names[::700]))

#%%

# Logistic regression with the new matrix?
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))



#%% md

# 2.) Stopwords

#%%

# Sklearn stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
print("Every 10th stop word:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))

#%%

# My own stop words
import pickle
extra_stopwords = pickle.load(open("extra_stopwords.p", "rb"))

#%%

# any differences?
# add them to the extra stopwords
extra_stopwords.extend([list(ENGLISH_STOP_WORDS)[i] for i in  [i for i,x in enumerate([i not in extra_stopwords for i in ENGLISH_STOP_WORDS]) if x]])
pickle.dump(extra_stopwords, open("extra_stopwords.p", "wb"))
print("Our stop words: {}".format((len(extra_stopwords))))
# More stopwords

#%%

# This one will be tested with extra stop words
X_train_extra = X_train
vect = CountVectorizer(min_df=5, stop_words="english" ).fit(text_train)
X_train = vect.transform(text_train)
print("X_train with stop words:\n{}".format(repr(X_train)))

#%%

# Now with out stop words
vect_extra = CountVectorizer(min_df=5, stop_words=frozenset(extra_stopwords) ).fit(text_train)
X_train_extra = vect_extra.transform(text_train)
print("X_train_extra with extra stop words:\n{}".format(repr(X_train_extra)))
# Much fewer elements in the vocabulary

#%%

# Grid search performance for the def X_train
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))

#%%

# Grid search performance for the def X_train_extra
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train_extra, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
# no difference


#%% md

# 3.) Rescaling the Data with tf-idf

#%%

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(TfidfVectorizer(min_df=5),
                     LogisticRegression())
param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("Best cross validation score: {:.2f}".format(grid.best_score_))
# No effect!

#%%

vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# Transform the training set
X_train = vectorizer.transform(text_train)
# Find maximum value for each of the features over the dataset
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# Get feature names
feature_names = np.array(vectorizer.get_feature_names())

print("Features with lowest tfidf:\n{}".format(feature_names[sorted_by_tfidf[:20]]))
print("Features with highest tfidf:\n{}".format(feature_names[sorted_by_tfidf[-20:]]))

#%%

sorted_by_idf = np.argsort(vectorizer.idf_)
print("Features with lowest idf:\n{}".format(
    feature_names[sorted_by_idf[:100]]
))

#%% md

# 4.) Model Coefficients

#%%

import mglearn
mglearn.tools.visualize_coefficients(
    grid.best_estimator_.named_steps["logisticregression"].coef_,
    feature_names,
    n_top_features=40
)

#%%

# 5.) n-Grams

#%%

pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
# running the grid search takes a long time because of the
# relatively large grid and the inclusion of trigrams
param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
"tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters:\n{}".format(grid.best_params_))

#%%

# extract scores from grid_search
scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
# visualize heat map
heatmap = mglearn.tools.heatmap(
scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
xticklabels=param_grid['logisticregression__C'],
yticklabels=param_grid['tfidfvectorizer__ngram_range'])
plt.colorbar(heatmap)


#%%

vect = grid.best_estimator_.named_steps['tfidfvectorizer']
feature_names = np.array(vect.get_feature_names())
coef = grid.best_estimator_.named_steps['logisticregression'].coef_
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)

#%%

# find 3-gram features
mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
# visualize only 3-gram features
mglearn.tools.visualize_coefficients(coef.ravel()[mask],
feature_names[mask], n_top_features=40)


#%% md

# 6.) Tokenization, Stemming, Lemmatization

#%% md

## 6.1.) Porter stemmer by spacy library

#%% md

Widely applied collection of heuristics

#%%

import nltk
import spacy
print('Spacy version: {}'.format(spacy.__version__))
import sklearn
print('nltk version: {}'.format(nltk.__version__))

#%%

# Spacy english lang.
en_nlp = spacy.load('en')
# instantiate nltk's Porter stemmer
stemmer = nltk.stem.PorterStemmer()

#%% md
# TODO: TRY THIS 2
### 6.1.2.) Compare Spacy's lemmatization and nltk's stemming

#%%

def compare_normalization(doc):
    # tokenize deoc in spacy
    doc_spacy = en_nlp(doc)
    # print lemmas found by spacy
    print("lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    # print tokens found by Porter stemmer
    print("stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])


#%%

compare_normalization(u"Our Meeting today was worse than yesterday,"
                      "I'm scared of meeting the clients tomorrow.")

#%% md
#
# Stemming is always restricted to trimming the word to a stem, so "was" becomes
# "wa", while lemmatization can retrieve the correct base verb form, "be". Similarly,
# lemmatization can normalize "worse" to "bad", while stemming produces "wors".
# Another major difference is that stemming reduces both occurrences of "meeting" to
# "meet". Using lemmatization, the first occurrence of "meeting" is recognized as a
# noun and left as is, while the second occurrence is recognized as a verb and reduced
# to "meet". In general, lemmatization is a much more involved process than stem‐
# ming, but it usually produces better results than stemming when used for normaliz‐
# ing tokens for machine learning.
#

#%%

# Technicality: we want to use the regexp-based tokenizer
# that is used by CountVectorizer and only use the lemmatization
# from spacy. To this end, we replace en_nlp.tokenizer (the spacy tokenizer)
# with the regexp-based tokenization.
import re
# regexp used in CountVectorizer
regexp = re.compile('(?u)\\b\\w\\w+\\b')

# load spacy language model
en_nlp = spacy.load("en", disable=['parser', 'ner'])
old_tokenizer = en_nlp.tokenizer
# replace the tokenizer with the preceding regexp
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(regexp.findall(string))

# create custom tokenizer using the spacy document processing pipeline
# (now using our own tokenizer)
def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    return [token.lemma_ for token in doc_spacy]
# define a count vectorizer with the custom tokenizer
lemma_vect = CountVectorizer(tokenizer = custom_tokenizer, min_df=5)

#%%

# transform text_train using CountVectorizer with lemmatization
#X_train_lemma = lemma_vect.fit_transform(text_train)
#print("X_train_lemma.shape: {}".format(X_train_lemma.shape))

# standard CountVectorizer for reference
vect = CountVectorizer(min_df=5).fit(text_train)
print("X_train.shape: {}".format(X_train.shape))

#%% md

# 7.) Topic Modelling

#%% md

# One particular technique that is often applied to text data is topic modeling, which is
# an umbrella term describing the task of assigning each document to one or multiple
# topics, usually without supervision.
#
# A good example for this is news data, which
# might be categorized into topics like “politics,” “sports,” “finance,” and so on.
#
# Each of the components we
# learn then corresponds to one topic, and the coefficients of the components in the
# representation of a document tell us how strongly related that document is to a par‐
# ticular topic. Often, when people talk about topic modeling, they refer to one particu‐
# lar decomposition method called Latent Dirichlet Allocation (often LDA for short).
#
# Intuitively, the LDA model tries to find groups of words (the topics) that appear
# together frequently. LDA also requires that each document can be understood as a
# “mixture” of a subset of the topics. It is important to understand that for the machine
# learning model a “topic” might not be what we would normally call a topic in every‐
# day speech,
#
#  Even if there is a semantic meaning for an LDA “topic”, it might not be some‐
# thing we’d usually call a topic. Going back to the example of news articles, we might
# have a collection of articles about sports, politics, and finance, written by two specific
# authors. In a politics article, we might expect to see words like “governor,” “vote,”
# “party,” etc., while in a sports article we might expect words like “team,” “score,” and
# “season.” Words in each of these groups will likely appear together, while it’s less likely
# that, for example, “team” and “governor” will appear together. However, these are not
# the only groups of words we might expect to appear together. The two reporters
# might prefer different phrases or different choices of words. Maybe one of them likes
# to use the word “demarcate” and one likes the word “polarize.” Other “topics” would
# then be “words often used by reporter A” and “words often used by reporter B,”
# though these are not topics in the usual sense of the word.
#
#
# #%% md
#
# **Let’s apply LDA to our movie review dataset to see how it works in practice. For
# unsupervised text document models, it is often good to remove very common words,
# as they might otherwise dominate the analysis. We’ll remove words that appear in at
# least 20 percent of the documents, and we’ll limit the bag-of-words model to the
# 10,000 words that are most common after removing the top 20 percent:**
#

#%%
# TODO: TRY IT 3
# TODO solved: gensim equivalent applied
vect = CountVectorizer(max_features=10000, max_df=.15)
X = vect.fit_transform(text_train)

#%% md

# We will learn a topic model with 10 topics, which is few enough that we can look at all
# of them. Similarly to the components in NMF, topics don’t have an inherent ordering,
# and changing the number of topics will change all of the topics.10 We’ll use the
# "batch" learning method, which is somewhat slower than the default ("online") but
# usually provides better results, and increase "max_iter", which can also lead to better
# models:


#%%

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10,    # Changed in version 0.19: n_topics was renamed to n_components
                               learning_method="batch",
                               max_iter=25,
                               random_state=0)
# We build the model and transform the data in one step
# Computing transform takes some time,
# and we can save time by doing both at once
document_topics = lda.fit_transform(X)

#%%

# how long goes it take?
import timeit
start = timeit.default_timer()
document_topics = lda.fit_transform(X)
stop = timeit.default_timer()

print('Time: ', stop - start)

#%%

print("lda.components_.shape: {}".format(lda.components_.shape))

#%%

# For each topic (a row in the components_), sort the features (ascending)
# Invert rows with [:, ::-1] to make sorting descending
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names from the vectorizer
feature_names = np.array(vect.get_feature_names())


#%%

# Print out the 10 topics:
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
sorting=sorting, topics_per_chunk=5, n_words=10)


#%% md

# What we are seeing above is after excluding the 20% of the commonly appearing words the most frequent appearing words in each topic group (the doc. LDA decided them to have the same topic). There is btw. not a single topic in a doc. but a collection of topics.

#%%

lda100 = LatentDirichletAllocation(n_components=100, learning_method="batch",
max_iter=25, random_state=0)
document_topics100 = lda100.fit_transform(X)
topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=topics, feature_names=feature_names,
sorting=sorting, topics_per_chunk=7, n_words=20)


#%%

# sort by weight of "music" topic 45
music = np.argsort(document_topics100[:, 45])[::-1]
# print the five documents where the topic is most important
for i in music[:10]:
# pshow first two sentences
  print(b".".join(text_train[i].split(b".")[:2]) + b".\n")

#%%

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
topic_names = ["{:>2} ".format(i) + " ".join(words)
for i, words in enumerate(feature_names[sorting[:, :2]])]
# two column bar chart:
for col in [0, 1]:
    start = col * 50
    end = (col + 1) * 50
    ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
    ax[col].set_yticks(np.arange(50))
    ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 2000)
    yax = ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()


#%% md

Topic models like LDA are interesting methods to understand large text corpora in
the absence of labels—or, as here, even if labels are available. The LDA algorithm is
randomized, though, and changing the random_state parameter can lead to quite
different outcomes. While identifying topics can be helpful, any conclusions you
draw from an unsupervised model should be taken with a grain of salt, and we rec‐
ommend verifying your intuition by looking at the documents in a specific topic. The
topics produced by the LDA.transform method can also sometimes be used as a com‐
pact representation for supervised learning. This is particularly helpful when few
training examples are available.


#%%


