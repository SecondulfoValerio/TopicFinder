#!/usr/bin/env python
# coding: utf-8

# In[49]:



#Imports
import os
import re
import nltk
import numpy as np
import tqdm
import spacy
import gensim
import pickle
from gensim.utils import simple_preprocess
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
import pandas as pd


# In[121]:


#FOR PAPERS
import os
directory = 'C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Business and project management\\DataCSV\\'
papers = pd.read_csv('C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Business and project management\\papers.csv')
papers = papers.drop(columns=['id', 'title', 'abstract','event_type', 'pdf_name', 'year'], axis=1)
papers = papers.sample(100) #limit to 100 papers due to the too large quantity of data to process

print(papers.head())


# In[120]:


#FOR PAPERS BUSINESS
import os
directory = 'C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Business and project management\\DataCSV\\'
papers = pd.read_csv('C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Business and project management\\DataCSV\\combined_file.csv')
papers = papers.iloc[:, 1:]
#papers = papers.drop(columns=['link', 'full_page','title', 'author','year', 'references', 'about'], axis=1)
papers = papers.sample(100) #limit to 100 papers due to the too large quantity of data to process

print(papers.head())


# In[3]:


#Creating Dataframe structure of scientifici papers and word count for each peaper
df_papers= pd.DataFrame(data=papers['paper_text'])
print(df_papers.head())

df_papers["word_count"]=df_papers['paper_text'].apply(lambda x: len(str(x).split(" ")))
print(df_papers[['paper_text','word_count']].head())


# In[4]:


df_papers.word_count.describe()


# In[5]:


#Identify the uncommon words

freq1= pd.Series(' '.join(df_papers['paper_text']).split()).value_counts()[-20:]
freq1


# In[6]:


#Text Preprocessing
corpus=[]
text_prep=[]
#Remove punctuations
    
papers['paper_text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))
#Convert to lower Case
    
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())
#Remove tags
    
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: re.sub('""',"",x))
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",x))

#Remove special characters and digits
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: re.sub("(\\d|\\W)+"," ",x))
#Removing links
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: re.sub('https:\/\/.*|http:\/\/.*|http.*',' ',x))                                                          

#simple preprocess will turn the documents into tokens in order to create a uniche list of words containing the words of all documents


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data = papers.paper_text_processed.values.tolist() #data preporccessed
data_words = list(sent_to_words(data)) #data_words contains in each row a preprocessed document splitted into list of words
print("n° preprocessed documents:", len(data_words))
print(data_words[:1][0][:30])



# In[ ]:





# In[7]:


#Bi-gram & Tri-gram
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # the more we increase threshold the fewer are the phraser retrieved
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[8]:


#Defining functions for Stop-words Bi-grams Tri-Grams and Lemmatization
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','using', 'show', 'result', 'large', 'iv', 'one', 'two', 'new', 'previously', 'shown','many','may','number','fig','image','however','et','al','xt','eus','aer','ib','omdp','yi','yt','hf','xt','ipt',
'also','log','zt'                  ])

def remove_stopwords(papers):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in papers]

def make_bigrams(papers):
    return [bigram_mod[doc] for doc in papers]

def make_trigrams(papers):
    return [trigram_mod[bigram_mod[doc]] for doc in papers]

def lemmatization(papers, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    papers_out = []
    for sent in papers:
        doc = nlp(" ".join(sent)) 
        papers_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        
    return papers_out,doc


# In[9]:


#Applying Stop-words Bi-grams Tri-grams creation and Lemmatization

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Create Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized,docsprep = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1][0][:30])


# In[10]:


#preprocessed documents rejointed

docs_prep=[]
for doc in data_words_nostops:
    docs_prep.append(" ".join(doc))
print(docs_prep)


# In[11]:


import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1][0][:30])


# In[12]:


#Exemples made on a single document of the texts preprocessed collection
#works on data preprocessed
#Word cloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=10,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(docs_prep[66]))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# In[13]:


#Exemples made on a single document of the texts preprocessed collection
#works on data lemmatized - changing the doc number we calculate the count vectorizer for that document
#we can do it for each document but for demostrative purposes and in order to save computational time i'll utilized a single document
cv= CountVectorizer(stop_words=list(stop_words),ngram_range=(1,3))
X=cv.fit_transform((data_lemmatized[1]))
list(cv.vocabulary_.keys())[:10]


# In[14]:


#Most frequently occuring words
def get_top_n_words(text,n=None):
    vec= CountVectorizer().fit(text)
    bag_of_words=vec.transform(text)
    sum_words= bag_of_words.sum(axis=0)
    words_freq=[(word,sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    words_freq= sorted(words_freq,key=lambda x: x[1],reverse=True)
    return words_freq[:n]


# In[15]:


#Convert most freq words to dataframe for plotting bar plot over set of documents
top_words=get_top_n_words(data_lemmatized[1],n=20)
top_df= pd.DataFrame(top_words)
top_df.columns=["Word","Frequency"]


# In[16]:


#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Frequency", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)


# In[17]:


#Most frequently occuring Bi-grams

def get_top_n2_words(text,n=None):
    vec1= CountVectorizer(ngram_range=(2,2), max_features=2000).fit(text)
    bag_of_words=vec1.transform(text)
    sum_words= bag_of_words.sum(axis=0)
    words_freq=[(word,sum_words[0,idx])for word,idx in vec1.vocabulary_.items()]
    words_freq=sorted(words_freq,key=lambda x:x[1], reverse=True)
    return words_freq[:n]


# In[18]:


#Bigram Plot
top2_words=get_top_n2_words(docs_prep,n=20)
top2_df=pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram","Freq"]
print(top2_df)

sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)


# In[19]:


#Tri-Grams
def get_top_n3_words(corpus,n=None):
    vec1=CountVectorizer(ngram_range=(3,3),max_features=2000).fit(corpus)
    bag_of_words=vec1.transform(corpus)
    sum_words= bag_of_words.sum(axis=0)
    words_freq=[(word,sum_words[0,idx]) for word, idx in vec1.vocabulary_.items()]
    words_freq=sorted(words_freq,key=lambda x:x[1],reverse=True)
    return words_freq[:n]


# In[20]:


top3_words= get_top_n3_words(docs_prep, n=20)
top3_df=pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram","Freq"]
print(top3_df)

sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram",y="Freq",data=top3_df)
j.set_xticklabels(j.get_xticklabels(),rotation=45)


# In[21]:


#Implementation of TF-IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)
#get feature names
feature_names=cv.get_feature_names_out()


# In[22]:


#ordering tf-idf results based on the score
def sort_coo(coo_matrix):
    tuples= zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples,key=lambda x: (x[1],x[0]), reverse=True)


# In[23]:


def extract_topn_from_vector(feature_names,sorted_items,topn=10):
    "get the feature names and tf-idf score of top n items"
    sorted_items= sorted_items[:topn]
    score_vals=[]
    feature_vals=[]
    #Index name and results
    for idx,score in sorted_items:
        score_vals.append(round(score,3))
        feature_vals.append(feature_names[idx])
        results={}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
    return results


# In[24]:



    #genera tf-idf sul set di documenti
    tf_idf_vector=tfidf_transformer.transform(cv.transform(docs_prep))
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #top n=10 items da restituire
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)

    print("\nKeywords:")
    for k in keywords:
        print(k,keywords[k])


# In[25]:


pip install -U gensim


# In[26]:


# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)


# In[299]:


#This whill show 10 topic with the most important words for each of them and the weight of the words in the topic contribution

from pprint import pprint

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[27]:


from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)


# In[28]:


# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='u_mass')
    
    return coherence_model_lda.get_coherence()


# In[305]:


#Now we'll proceed to find the best hyperparameters for LDA model and in order for that we'll use a function that will calculate
#the various combination of hyperparameters and choerence score results
#for this i've decided to use u_mass scoring function instead of c_v due to the very very high time and resources that this function
#required in order to work beeing u_mass way faster
grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
               corpus]

corpus_title = ['75% Corpus', '100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Data Mining\\Slides\\lda_tuning_results.csv', index=False)
    pbar.close()


# In[33]:


#plot the hyperparameters values obtained
plot_res = pd.read_csv('C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Data Mining\\Slides\\lda_tuning_results.csv')
print(plot_res.columns)
import matplotlib.pyplot as plt

# Estrai le colonne necessarie dal DataFrame
x = plot_res['Topics']
y = plot_res['Coherence']

# Crea il grafico a dispersione
plt.scatter(x, y)

# Aggiungi etichette agli assi e un titolo al grafico
plt.xlabel('topics')
plt.ylabel('coherence')
plt.title('Topic/Coherence score')

# Mostra il grafico
plt.savefig('C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Data Mining\\Slides\\TopicCoherenceGraph.png')
plt.show()


# In[37]:


#We try now to choose only 2-3 topics and see the optimal parameters
interested_rows = plot_res[plot_res['Topics'] <= 3] #we choose only the rows having 3 or less than 3 topics
print(interested_rows)

x_3 = plot_res['Topics']
y_3 = plot_res['Coherence']

# Crea il grafico a dispersione
plt.scatter(x_3, y_3)

# Aggiungi etichette agli assi e un titolo al grafico
plt.xlabel('topics')
plt.ylabel('coherence')
plt.title('Topic/Coherence score')

# Mostra il grafico
plt.savefig('C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Data Mining\\Slides\\TopicCoherenceGraphFor3Topics.png')
plt.show()


# In[45]:


#Select the rows with coherence score under -0.28 , the closests to 0
interested_rows_zero = interested_rows[interested_rows['Coherence'] >= -0.28 ] #we choose only the rows having 3 or less than 3 topics
print(interested_rows_zero)
#now we consider the best parameters for the min coherence values of 2 and 3 topics
interested_rows_min2 = interested_rows_zero[interested_rows_zero['Topics']==2 ].min() #we choose only the rows having 3 or less than 3 topics
print("FOR 2 TOPICS")
print(interested_rows_min2)
#now we do the same for 3 topics
interested_rows_min2 = interested_rows_zero[interested_rows_zero['Topics']==3 ].min() #we choose only the rows having 3 or less than 3 topics
print("FOR 3 TOPICS")
print(interested_rows_min2)


# In[65]:


#Setting LDA model with 2 topics parameters
num_topics = 2

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=0.01,
                                           eta=0.01)


# In[68]:


import pyLDAvis.gensim_models as gensimvis
import pickle 
import pyLDAvis

# Visualize the topics
#Saliency, is the measure of how much the term tells you about that topic
#Relevance, is a measure of the importance of the word for that topic, compared to the rest of the topics.
pyLDAvis.enable_notebook()

LDAvis_data_filepath = os.path.join('C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Data Mining\\Slides\\lda_model.csv'+str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, 'C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Data Mining\\Slides\\lda_model.csv'+ str(num_topics) +'.html')

LDAvis_prepared


# In[69]:


#Setting LDA model with 3 topics parameters
num_topics = 3

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=0.01,
                                           eta=0.01)


# In[70]:


# Visualize the 3 topics results
pyLDAvis.enable_notebook()

LDAvis_data_filepath = os.path.join('C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Data Mining\\Slides\\lda_model.csv'+str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, 'C:\\Users\\Rauro\\OneDrive\\Desktop\\Uni\\Data Mining\\Slides\\lda_model.csv'+ str(num_topics) +'.html')

LDAvis_prepared


# In[ ]:





# In[79]:


#For last we try LSA
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel

#Preparation
dictionary=id2word
doc_term_matrix = [dictionary.doc2bow(doc) for doc in data_lemmatized]


# In[80]:


def create_gensim_lsa_model(doc_prep,number_of_topics,words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
 
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel


# In[99]:


n_topics_lsa=10 #try 10 topics
lsamodel= create_gensim_lsa_model(data_lemmatized,n_topics_lsa,10) #10 words per topic


# In[92]:


def compute_coherence_values(dictionary, doc_term_matrix, doc_prep, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=n_topics_lsa, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_prep, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


# In[93]:


def plot_graph(doc_prep,start, stop, step):
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_prep,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


# In[94]:


start,stop,step=2,12,1
plot_graph(data_lemmatized,start,stop,step)


# In[100]:


lsamodel= create_gensim_lsa_model(data_lemmatized,n_topics_lsa,3) #10 words per topic


# In[111]:


i=0
topic=[]
while i<=3
    st= lsamodel.print_topic(0)
    st.
    topic.append(lsamodel.print_topic(0))
print(topics)  # Linea vuota per separare le tuple


# In[112]:


lsamodel.show_topics(num_topics=-1, num_words=10, log=False, formatted=True)

