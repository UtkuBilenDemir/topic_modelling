import pandas as pd
import pickle
import pprint
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import np
import matplotlib as plt
import os

import re
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

os.getcwd()
# 1. Get the dataset

knoc_df = pd.read_csv("./data/knoc_AB.csv")
knoc_df.head()

knoc_df.columns
knoc_df.shape
knoc_df = knoc_df.fillna("nan")

# unpack all the individual research areas
## res_areas = []
## nested_res_areas = [i.split(";") for i in knoc_df['web.of.science.categories.']]
## while nested_res_areas:
##     res_areas.extend(nested_res_areas.pop(0))
## res_areas = [i.strip().replace("(|)", "") for i in res_areas]
## res_areas = set(res_areas)
## list(enumerate(res_areas))

## pickle.dump(list(res_areas), open("../data/res_areas.p", "wb"))
res_areas = pickle.load(open("./data/res_areas.p", "rb"))
list(enumerate(res_areas))
res_areas[22]

selected_ra_indexes = [1, 7, 10, 13, 17,
                       22, 25, 47, 56, 58,
                       63, 70, 74, 91, 93,
                       99, 100, 102, 103, 240,
                       116, 118, 128, 135, 142,
                       152, 165, 166, 167, 187,
                       248]
selected_res_areas =[res_areas[i] for i in selected_ra_indexes]
selected_res_areas

ra_matched_pubs = [i in selected_res_areas for i in knoc_df["web.of.science.categories."]]
sum(ra_matched_pubs)
# 5650 publications

ra_matched_pub_indexes = [i for i,x in enumerate(ra_matched_pubs) if x == True ]

import collections
collections.Counter([knoc_df["web.of.science.categories."][i] for i in ra_matched_pub_indexes])

# 2. Prepare the abstracts and keywords from the selected research areas

# new df that only includes the selected pubs
sel_knoc_df = knoc_df.iloc[ra_matched_pub_indexes,:]
sel_knoc_df = sel_knoc_df.reset_index()
sel_knoc_df.shape
sel_knoc_df["TI"][1432]
# Unify the abstrats with the keywords in another column
## unified_abs_key = [sel_knoc_df["AB"][i] + sel_knoc_df["TI"] + sel_knoc_df["DE"][i] + sel_knoc_df["ID"][i] for i in range(sel_knoc_df.shape[0])]
sel_knoc_df["web.of.science.categories."][2000:2010]
pprint(set([i.title().replace("\\", "") for i in sel_knoc_df["web.of.science.categories."]]))

unified_abs_key = [""] * len(sel_knoc_df)
len(unified_abs_key)
for i in range(len(sel_knoc_df)):
    unified_abs_key[i] = sel_knoc_df["AB"][i] +  sel_knoc_df["DE"][i] + sel_knoc_df["ID"][i]
    print(i)

unified_abs_key = [unified_abs_key[i] + sel_knoc_df["TI"][i] for i in range(len(unified_abs_key))]
sel_knoc_df["web.of.science.categories."]
sel_knoc_df["TI"][2]
sum([len(i) for i in unified_abs_key]) / len([len(i) for i in unified_abs_key])
unified_abs_key[2006]



# --------------------------------------- i.) Import stop words
stopwords_extra = pickle.load(open("./ext_sources/extra_stopwords.p", "rb"))
stopwords_extra.append("china")
stopwords_extra.append("chinese")
stopwords_extra.append("nannannan")
stopwords_extra.append("elsevier")

## # Book
## # ---
## # ii.) Create a vocabulary
## vect = CountVectorizer().fit(unified_abs_key)
## vect.get_feature_names()
## # iii.) Create a document-term matrix
## abs_key_dtmatrix = vect.transform(unified_abs_key)
## print("abs_key_dtmatrix:\n{}".format(repr(abs_key_dtmatrix)))
## # The vocabulary contains 74849 elements, thus the number of rows = 74849
## # Columns represent the documents
##
## # Another way to access vocabulary
## feature_names = vect.get_feature_names()
## print("Number of features: {}".format(len(feature_names)))
## print("First 20 features:\n{}".format(feature_names[:20]))
## print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
## print("Every 2000th feature:\n{}".format(feature_names[::2000]))
##
## vect = CountVectorizer(min_df=5).fit(text_train)
## X_train = vect.transform(text_train)
## print("X_train with min_df=5: {}".format(repr(X_train)))
## # ---
## # Book
## print(vect.get_params())

# --------------------------------------- Coh
# --------------------------------------- ---
# --------------------------------------- ii.) Tokenize the docs
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
token_abs_key = list(sent_to_words(unified_abs_key))
token_abs_key[6]
## ubd: This one is definitely a better introduction

# --------------------------------------- iii.) Remove stopwords
# Define function for stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords_extra] for doc in texts]
# Remove Stop Words
token_abs_key_nostop = remove_stopwords(token_abs_key)
print("With stop words: {} words\nWithout stop words: {} words\nDiscarded words: {}".format(len(token_abs_key[6]),
                                                                                            len(token_abs_key_nostop[6]),
                                                                                            [i for i in token_abs_key[6] if i  not in token_abs_key_nostop[6]]))
## [print(i + "|" + j) for i, j in zip(token_abs_key[6], token_abs_key_nostop[6])]
## print("\n".join("{} || {}".format(x, y) for x, y in zip(token_abs_key[6], token_abs_key_nostop[6])))


# --------------------------------------- iv.) Form Bigrams
bigram = gensim.models.Phrases(token_abs_key_nostop, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
abs_key_bigrams = make_bigrams(token_abs_key_nostop)
# TODO: Here are some duplicates, what's up with those??
# TODO solved: Those appear a few times on the text, nothing to worry about
len(abs_key_bigrams[6])

trigram = gensim.models.Phrases(bigram[abs_key_bigrams], threshold=100)
trigram_mod = gensim.models.phrases.Phraser(trigram)
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
abs_key_trigrams = make_trigrams(abs_key_bigrams)


len(abs_key_trigrams[6])
abs_key_trigrams = [[word for word in simple_preprocess(str(doc)) if word != "rights_reserved" ] for doc in abs_key_trigrams]

len(abs_key_trigrams[6])


# --------------------------------------- v.) Lemmatize
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
nlp = spacy.load('en', disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
# TODO: IT MAKES THINKGS WORSE?????
# TODO: MORE INFO ABOUT LEMMAt.
# TODO: THE ORDER IS IMPORTANT FOR LDA?
old_lemma = lemmatization(abs_key_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


##
# Lemmatize the documents.
#

#
from nltk.stem.wordnet import WordNetLemmatizer
#

#
# Lemmatizer 2 
#
lemmatizer = WordNetLemmatizer()
#
abs_key_lemmatized = [[lemmatizer.lemmatize(token) for token in doc] for doc in abs_key_trigrams]
#



sum([i == "BECOME" for i in stopwords_extra])
### 
### unified_abs_key[2003]
### ### old_ lemma = abs_key_lemmatized
### old_lemma[2003]
### ### new_lemma = abs_key_lemmatized
### new_lemma[2003]
### 


# --------------------------------------- 2. Create the Dictionary and Corpus needed for Topic Modeling
# Create Dictionary
# TODO: WHICH ONE IS BETTER
id2word = corpora.Dictionary(abs_key_lemmatized)
### id2word = corpora.Dictionary(abs_key_bigrams)
# TODO: Better? We remove rare words and common words based on their document frequency.
#  Below we remove words that appear in less than 20 documents or in more than 50% of the documents.
#  Consider trying to remove words only based on their frequency, or maybe combining that with this approach.
id2word.filter_extremes( no_above=0.25)


# Create Corpus
texts = abs_key_lemmatized
### texts = abs_key_bigrams

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus)

# Gensim creates a unique id for each word in the document. The produced corpus shown above is a mapping of (word_id, word_frequency).
#
# For example, (0, 1) above implies, word id 0 occurs once in the first document. Likewise, word id 1 occurs twice and so on.
#
# This is used as the input by the LDA model.
#
# If you want to see what word a given id corresponds to, pass the id as a key to the dictionary.

id2word[327]

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in [corpus[6]]]

# --------------------------------------- 3. Building the Topic Model
# TODO: Discard this, we will run coh. estimation anyways
# TODO: EXPLAIN EVERY SINGLE PARAMETER
# --------------------------------------- Build LDA model
## lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
##                                            id2word=id2word,
##                                            num_topics=20,
##                                            random_state=100,
##                                            update_every=1,
##                                            chunksize=100,
##                                            passes=10,
##                                            alpha='auto',
##                                            per_word_topics=True)
##
## # --------------------------------------- 4. View the topics in LDA model
## # Print the Keyword in the 10 topics
## pprint(lda_model.print_topics())
## doc_lda = lda_model[corpus]
##
## # Compute Perplexity
## print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
##
## # Compute Coherence Score
## coherence_model_lda = CoherenceModel(model=lda_model, texts=abs_key_lemmatized, dictionary=id2word, coherence='c_v')
## coherence_lda = coherence_model_lda.get_coherence()
## print('\nCoherence Score: ', coherence_lda)
##
corpus
# --------------------------------------- 5. Building LDA Mallet Model INSTEAD
mallet_path = './ext_sources/dependencies/mallet-2.0.8/bin/mallet' # update this path
# TODO: TRY DIFFERENT CORPORA
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=abs_key_lemmatized, dictionary=id2word, coherence='c_v')
### coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=abs_key_bigrams, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)



# --------------------------------------- 6. Find the optimal number of topics

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

import timeit
start = timeit.default_timer()
# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=abs_key_lemmatized, start=2, limit=40, step=2)
### model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=abs_key_bigrams, start=2, limit=40, step=4)
stop = timeit.default_timer()
print('Time: ', stop - start)

# Show graph
limit=40; start=2; step=2;
x = range(start, limit, step)

coherence_values[10] =0.5840828
coherence_values[11] =0.5745439

plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

import plotly.graph_objects as go
import numpy as np
fig = go.Figure(data=go.Scatter(x=list(x), y=coherence_values, line=dict(color="#DA4E52")))
fig.update_layout( # plot_bgcolor="#F0F1EB",
                   paper_bgcolor='rgba(0,0,0,0)',
                   # plot_bgcolor='rgba(0,0,0,0)',
                   xaxis_title='Num. of Topics',
                   yaxis_title='Coherence score')
fig.show()
fig.write_html("./presentation/reveal.js/visualizations/coherence.html")
# --------------------------------------- 7. Select the model and print the topics
optimal_model = model_list[10]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# --------------------------------------- 8. Visualize the topics
import pyLDAvis
import pyLDAvis.gensim as gensimvis
pyLDAvis.disable_notebook()
optimal_model_gensim = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model)
vis = pyLDAvis.gensim.prepare(optimal_model_gensim, corpus, id2word)
# pyLDAvis.show(vis)
# pyLDAvis.prepared_data_to_html(vis)
pyLDAvis.save_html(vis,"test11_oldlemma.html")



list(optimal_model.load_document_topics())[2003]
optimal_model.print_topics()

unified_abs_key[2003]