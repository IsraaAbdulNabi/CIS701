import normalise
import string
import re
import pandas as pd

import string
def punctuation_remove(s):
    table = str.maketrans({key: None for key in string.punctuation})
    return s.translate(table)

def word_count(train_df):
 train_df['word_count'] = train_df["Text"].str.lower().str.split().apply(len)
 print(train_df.head())

def preProcess(train_lines,test_lines):
    # Convert from raw binary strings to strings that can be parsed
    train_file_lines = [x.decode('utf-8') for x in train_lines]
    test_file_lines = [x.decode('utf-8') for x in test_lines]

    train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
    train_labels = train_labels[:100000]
    train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]
    train_sentences=train_sentences[:100000]

    for i in range(len(train_sentences)):
        train_sentences[i] = re.sub('\d', '', train_sentences[i])

    test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]
    test_labels=test_labels[:50000]
    test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]
    test_sentences=test_sentences[:50000]

    for i in range(len(test_sentences)):
        test_sentences[i] = re.sub('\d', '', test_sentences[i])

    for i in range(len(train_sentences)):
        if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in \
                train_sentences[i]:
            train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

    for i in range(len(test_sentences)):
        if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in \
                test_sentences[i]:
            test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

    train_df = pd.DataFrame(columns=['Text', 'Sentiment', 'WordCount'])
    train_df['Text'] = train_sentences
    train_df['Sentiment'] = train_labels
    train_df['WordCount'] = word_count(train_df)
    test_df = pd.DataFrame(columns=['Text', 'Sentiment', 'WordCount'])
    test_df['Text'] = test_sentences
    test_df['Sentiment'] = test_labels
    test_df['WordCount'] = word_count(test_df)

    print("Proportion of positive review:", len(train_df[train_df.Sentiment == 1]) / len(train_df))
    print("Proportion of negative review:", len(train_df[train_df.Sentiment == 0]) / len(train_df))
    # Remove all punctuations
    train_df['Text'] = train_df['Text'].apply(punctuation_remove)
    test_df['Text'] = test_df['Text'].apply(punctuation_remove)

    #Normalise Module takes a text as input, and returns it in a normalised form, ie. expands all word tokens deemed not to be of a standard type.
    #train_sentences = normalise(train_sentences, verbose=True)
    #test_sentences = normalise(test_sentences, verbose=True)


    return train_df, test_df