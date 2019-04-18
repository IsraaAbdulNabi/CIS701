import nltk
nltk.download('brown')
nltk.download('names')
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import pandas as pd
import gc
import time
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

def NLTK_Model(train_df,test_df ):
 # Create train and test dataframes
 Na_train = {'Sentence': train_df["Text"], 'Label': train_df['Sentiment']}
 Nav_train = pd.DataFrame(Na_train)

 Na_test = {'Sentence': test_df["Text"], 'Label': test_df['Sentiment']}
 Nav_test = pd.DataFrame(Na_test)

 Nav_train.head()

 Nav_train = Nav_train.head(900)
 Nav_test = Nav_test.head(100)

#Separate Positive and Negative reviews

 del Na_train, Na_test
 gc.collect()

 #Cleaning and Feature Extraction
 sents = []
 alll = []
 stopwords_set = set(stopwords.words("english"))

 for index, row in Nav_train.iterrows():
    words_filtered = [e.lower() for e in row.Sentence.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    sents.append((words_without_stopwords, row.Label))
    alll.extend(words_without_stopwords)

 # Extracting word features def get_words_in_reviews(reviews): alll = [] for (words, sentiment) in reviews: alll.extend(words) return alll
 def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
 w_features = get_word_features(alll)
 # TESTING BELOW
 def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

 # Training the Naive Bayes classifier
 start_time1 = time.time()
 training_set = nltk.classify.apply_features(extract_features,sents)

 classifier = nltk.NaiveBayesClassifier.train(training_set)
 #train_pos = Nav_train[Nav_train['Label'] == 1]
 #train_pos = train_pos['Sentence']
 #train_neg = Nav_train[Nav_train['Label'] == 0]
 #train_neg = train_neg['Sentence']
 test_pos = Nav_test[Nav_test['Label'] == 1]
 test_pos = test_pos['Sentence']
 test_neg = Nav_test[Nav_test['Label'] == 0]
 test_neg = test_neg['Sentence']

 test_neg.head(40)
 neg_cnt = 0
 pos_cnt = 0
 for obj in test_neg:
    res = classifier.classify(extract_features(obj.split()))
    if (res == 0):
        neg_cnt = neg_cnt + 1
 for obj in test_pos:
    res = classifier.classify(extract_features(obj.split()))
    if (res == 1):
        pos_cnt = pos_cnt + 1

 NLTK_Exc_Time = time.time() - start_time1;
 print('[Negative]: %s/%s ' % (len(test_neg), neg_cnt))
 print('[Positive]: %s/%s ' % (len(test_pos), pos_cnt))
 #aa = classifier.evaluate(Nav_test['Sentence'],Nav_test['Label'])
 acccc= ((neg_cnt+pos_cnt)/(len(test_neg)+len(test_pos))) * 100
 print("Accuracy by nltk classifier is", acccc)

 print(test_neg.loc[52])
 classifier.classify(extract_features(test_neg.loc[52].split()))

 return NLTK_Exc_Time