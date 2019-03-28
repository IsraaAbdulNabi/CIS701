import normalise
import string

def toLower(text):
    text = text.lower()
    return text


def removePunctuation(text):
    text = text.translate(string.maketrans("", ""), string.punctuation)
    return text

def normaliseText(text):
    """
    This module takes a text as input, and returns it in a normalised form, ie. expands all word tokens deemed not to be of a standard type. Non-standard words (NSWs) are detected, classified and expanded. Examples of NSWs that are normalised include:
    Numbers - percentages, dates, currency amounts, ranges, telephone numbers.
    Abbreviations and acronyms.
    Web addresses and hashtags.

    :param text:input text can be a list of words, or a string.
    :return:
    """
    text=normalise(text, verbose=True)
    return text
