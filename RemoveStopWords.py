import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  tokens = word_tokenize(text)
  result = [i for i in tokens if not i in stop_words]
  print (result)
  return result