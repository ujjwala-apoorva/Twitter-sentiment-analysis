#!/usr/bin/env python
# coding: utf-8

# # COVID Twitter Sentiment Analysis 

# This project is about the analysis of tweets about coronavirus, with the goal of performing a Sentiment Analysis using BERT Model

# ## Importing Libraries

# We are importing different libraries and models for Different uses. Below are the models we would be using in our pipeline. We have segregated them and imported according their of group.

# In[104]:


#general purpose packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

#data processing
import re, string
import emoji
import nltk

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel


#keras
import tensorflow as tf
from tensorflow import keras


#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

#set seed for reproducibility
seed=42

#set style for plots
sns.set_style("whitegrid")
sns.despine()
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)


# ## Loading the data set

# The data set has been taken from kaggle , and the data set was already seperated between train and test data set, hence we would be reading both of them seperately, under the name df and test_df respectively.

# In[29]:


df = pd.read_csv('Corona_NLP_train.csv',encoding='ISO-8859-1')
df_test = pd.read_csv('Corona_NLP_test.csv')


# NOTE: UTF-8 encoding does not work on the dataset when loading it with pandas 'read_csv' function. This lead to the use of 'ISO-8859-1'/latin-1 encoding.
# It will be found later that some special characters like apostrophes are turned into '\x92', which will be taken care of during the data cleaning process.

# In[30]:


df.head(3)


# In[31]:


df.info()


# We convert the date column 'TweetAt' to pandas datetime format to improve its usability in the further analysis.

# In[32]:


df['TweetAt'] = pd.to_datetime(df['TweetAt'])


# # Data Preprocessing

# ## Removing duplicates

# In[33]:


df.drop_duplicates(subset='OriginalTweet',inplace=True)


# In[34]:


df.info()


# We find that there were no duplicate tweets, but it is a good step to follow anuways for better results.

# # Data visualization

# Now we would be calculating and visualising the data set according to the number of Tweets count per day, to understand the number of tweets made each day.
# 

# In[36]:


tweets_per_day = df['TweetAt'].dt.strftime('%m-%d').value_counts().sort_index().reset_index(name='counts')


# In[37]:


plt.figure(figsize=(20,5))
ax = sns.barplot(x='index', y='counts', data=tweets_per_day,edgecolor = 'black',ci=False, palette='Blues_r')
plt.title('Tweets count by date')
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()


# We notice that in the dataset there are some days without tweets in the dataset. Among the days with tweets, most of them are made around the end of March: from 18th of Match to the 26th of March.
# 
# 

# Now, we would try to visualise which Country or City had the most contribution in making tweets

# In[39]:


tweets_per_country = df['Location'].value_counts().loc[lambda x : x > 100].reset_index(name='counts')


# In[40]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x='index', y='counts', data=tweets_per_country,edgecolor = 'black',ci=False, palette='Spectral')
plt.title('Tweets count by country')
plt.xticks(rotation=70)
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()


# The 'location' column contains both countries and cities. It could be interesting to separate cities and countries, however this wont be investigated in this work.
# 
# 

# # Countinue with deep data cleaning

# In the following, we will perform some data cleaning on the raw text of the tweets.
# To simplify the analaysis, we will just keep the columns 'Originaltweet' (raw tweets) and the target column 'Sentiment

# In[41]:


df = df[['OriginalTweet','Sentiment']]


# In[42]:


df_test = df_test[['OriginalTweet','Sentiment']]


# Then we define custom functions to clean the text of the tweets.

# In[43]:


##CUSTOM DEFINED FUNCTIONS TO CLEAN THE TWEETS

#Clean emojis from text
def strip_emoji(text):
    return re.sub(emoji.get_emoji_regexp(), r"", text) #remove emoji

#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text


# In[44]:



#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2


# In[45]:


#Filter special characters such as & and $ present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)


# In[46]:


texts_new = []
for t in df.OriginalTweet:
    texts_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(t))))))


# In[47]:


texts_new_test = []
for t in df_test.OriginalTweet:
    texts_new_test.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(t))))))


# Now we can create a new column, for both train and test sets, to host the cleaned version of the tweets' text.
# 
# 

# In[50]:


df['text_clean'] = texts_new
df_test['text_clean'] = texts_new_test


# In[51]:


df['text_clean'].head()


# In[52]:


df_test['text_clean'].head()


# In[53]:


df['text_clean'][1:8].values


# Moreover, we will also create a column to host the lenght of the cleaned text, to check if by cleaning the text we removed too much text or almost entirely the tweet!
# 
# 

# In[54]:


text_len = []
for text in df.text_clean:
    tweet_len = len(text.split())
    text_len.append(tweet_len)


# In[55]:


df['text_len'] = text_len


# In[56]:


text_len_test = []
for text in df_test.text_clean:
    tweet_len = len(text.split())
    text_len_test.append(tweet_len)


# In[57]:


df_test['text_len'] = text_len_test


# In[58]:


plt.figure(figsize=(7,5))
ax = sns.countplot(x='text_len', data=df[df['text_len']<10], palette='mako')
plt.title('Training tweets with less than 10 words')
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()


# In[59]:


plt.figure(figsize=(7,5))
ax = sns.countplot(x='text_len', data=df_test[df_test['text_len']<10], palette='mako')
plt.title('Test tweets with less than 10 words')
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()


# As we can see, there are lots of cleaned tweets with 0 words: this is due to the cleaning performed before. This means that some tweets contained only mentions, hashtags and links, which have been removed. We will drop these empty tweets and also those with less than 5 words. By doing that we would be removin the outliers and the not so important data, which can be removed for now for better prediction

# In[60]:


print(f" DF SHAPE: {df.shape}")
print(f" DF TEST SHAPE: {df_test.shape}")


# In[61]:


df = df[df['text_len'] > 4]


# In[62]:


df_test = df_test[df_test['text_len'] > 4]


# In[63]:


print(f" DF SHAPE: {df.shape}")
print(f" DF TEST SHAPE: {df_test.shape}")


# ## Training data deeper cleaning

# Let's perform a further cleaning checking the tokenizer version of the sentences using BERT
# 
# 

# In[65]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# In[67]:


token_lens = []

for txt in df['text_clean'].values:
    tokens = tokenizer.encode(txt, max_length=512, truncation=True)
    token_lens.append(len(tokens))
    
max_len=np.max(token_lens)


# In[68]:


print(f"MAX TOKENIZED SENTENCE LENGTH: {max_len}")


# Let's check the long tokenized sentences (with more than 80 tokens ):
# 
# 

# In[69]:


token_lens = []

for i,txt in enumerate(df['text_clean'].values):
    tokens = tokenizer.encode(txt, max_length=512, truncation=True)
    token_lens.append(len(tokens))
    if len(tokens)>80:
        print(f"INDEX: {i}, TEXT: {txt}") 


# These sentences are not in english. They should be dropped.
# 
# 

# In[70]:


df['token_lens'] = token_lens


# In[71]:


df = df.sort_values(by='token_lens', ascending=False)
df.head(20)


# In[72]:


df = df.iloc[12:]
df.head()


# The dataset looks more clean now. We will shuffle it and reset the index.
# 
# 

# In[73]:


df = df.sample(frac=1).reset_index(drop=True)


# ## Test data deeper cleaning
# 

# We will perform the data cleaning based on the tokenized sentences on the test set.
# 
# 

# In[74]:


token_lens_test = []

for txt in df_test['text_clean'].values:
    tokens = tokenizer.encode(txt, max_length=512, truncation=True)
    token_lens_test.append(len(tokens))
    
max_len=np.max(token_lens_test)


# In[75]:


print(f"MAX TOKENIZED SENTENCE LENGTH: {max_len}")


# In[76]:


token_lens_test = []

for i,txt in enumerate(df_test['text_clean'].values):
    tokens = tokenizer.encode(txt, max_length=512, truncation=True)
    token_lens_test.append(len(tokens))
    if len(tokens)>80:
        print(f"INDEX: {i}, TEXT: {txt}")


# In[77]:


df_test['token_lens'] = token_lens_test


# In[78]:


df_test = df_test.sort_values(by='token_lens', ascending=False)
df_test.head(10) 


# In[79]:


df_test = df_test.iloc[5:]
df_test.head(3)


# In[80]:


df_test = df_test.sample(frac=1).reset_index(drop=True)


# Now the data cleaning is completed. I will perform more data cleaning if I have new ideas !! :)
# 
# 

# ## Sentiment column analysis
# 

# Now we will look at the target column 'Sentiment'.
# 
# 

# In[82]:


df['Sentiment'].value_counts()


# The first thing we can do is to encode the categories with numbers. We will also create just 3 possible emotions: Positive, Neutral and Negative.
# 
# 

# In[83]:


df['Sentiment'] = df['Sentiment'].map({'Extremely Negative':0,'Negative':0,'Neutral':1,'Positive':2,'Extremely Positive':2})


# In[84]:


df_test['Sentiment'] = df_test['Sentiment'].map({'Extremely Negative':0,'Negative':0,'Neutral':1,'Positive':2,'Extremely Positive':2})


# In[85]:


df['Sentiment'].value_counts()


# We note that the three classes are imbalanced. We will proceed with oversampling the train test, to remove bias towards the majority classes.
# 
# 

# ### Class Balancing by RandomOverSampler
# 

# As we saw above, that our classes were highly imbalanced. We would be using the RandomOverSampler, to remove the bias towards the majority class by creating  more values and data.

# In[86]:


ros = RandomOverSampler()
train_x, train_y = ros.fit_resample(np.array(df['text_clean']).reshape(-1, 1), np.array(df['Sentiment']).reshape(-1, 1));
train_os = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['text_clean', 'Sentiment']);


# In[87]:


train_os['Sentiment'].value_counts()


# ## Train - Validation - Test split
# 

# In[88]:


X = train_os['text_clean'].values
y = train_os['Sentiment'].values


# A validation set will be extracted from the training set to monitor the validation accuracy, and so prevent overfitting.
# 
# 

# In[89]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)


# In[90]:


X_test = df_test['text_clean'].values
y_test = df_test['Sentiment'].values


# ## One hot encoding
# 

# By using one hot encoding on the target variable we achieved higher accuracy. For this reason we will choose one hot enconding over label encoding.
# 
# 

# In[91]:


ohe = preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()
y_test = ohe.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()


# In[92]:


print(f"TRAINING DATA: {X_train.shape[0]}\nVALIDATION DATA: {X_valid.shape[0]}\nTESTING DATA: {X_test.shape[0]}" )


# ## BERT Sentiment Analysis
# 

# ### BERT Tokenization
# 

# We already performed a basic analyis of the tokenized sentences, now we just need to define a custom tokenizer function and call the encode_plus method of the BERT tokenizer.
# 
# 

# In[93]:


MAX_LEN=128


# In[94]:


def tokenize(data,max_len=MAX_LEN) :
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)


# Then, we apply the tokenizer function to the train, validation and test sets.
# 
# 

# In[95]:


train_input_ids, train_attention_masks = tokenize(X_train, MAX_LEN)
val_input_ids, val_attention_masks = tokenize(X_valid, MAX_LEN)
test_input_ids, test_attention_masks = tokenize(X_test, MAX_LEN)


# ### BERT modeling
# 

# Now we can import the BERT model from the pretrained library from Hugging face.
# 
# 

# In[96]:


bert_model = TFBertModel.from_pretrained('bert-base-uncased')


# Then, we create a custom function to host the pre trained BERT model, and attach to it a 3 neurons output layer, necessary to perform the classification of the 3 different classes of the dataset (the 3 emotions).
# 
# 

# In[97]:


def create_model(bert_model, max_len=MAX_LEN):
    
    ##params###
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()


    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    embeddings = bert_model([input_ids,attention_masks])[1]
    
    output = tf.keras.layers.Dense(3, activation="softmax")(embeddings)
    
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks], outputs = output)
    
    model.compile(opt, loss=loss, metrics=accuracy)
    
    return model


# In[98]:


model = create_model(bert_model, MAX_LEN)
model.summary()


# Finally we can start fine tuning the BERT transformer !
# 
# 

# In[99]:


history_bert = model.fit([train_input_ids,train_attention_masks], y_train, validation_data=([val_input_ids,val_attention_masks], y_valid), epochs=3, batch_size=32)


# ### BERT results
# 

# In[100]:


result_bert = model.predict([test_input_ids,test_attention_masks])


# In[101]:


y_pred_bert =  np.zeros_like(result_bert)
y_pred_bert[np.arange(len(y_pred_bert)), result_bert.argmax(1)] = 1


# In[102]:


conf_matrix(y_test.argmax(1), y_pred_bert.argmax(1),'BERT Sentiment Analysis\nConfusion Matrix')


# In[103]:


print('\tClassification Report for BERT:\n\n',classification_report(y_test,y_pred_bert, target_names=['Negative', 'Neutral', 'Positive']))


# The preformance of the Algorithm went quite well on the dataset, showing F1 and accuracy scores around 90%.
# Such high scores can only be achieved when a a good cleaning of the original data has been done, allowing the algorithms to learn the most from it.
# 
# The training took around 27 minutes per epoch (for a total of 3 epochs) per algorithm, since both the transformers parameters (more than 100 million) have been fine tuned to perform the best on the given dataset. It is possible to train only the last layer of the transformer without fine tuning the other parameters: however, this usually lead to inferior results compared to the fine tuning approach.
# 
# 

# In[ ]:




