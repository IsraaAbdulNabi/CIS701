# pip install spacy
# python -m spacy download en_core_web_sm

import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")
def tokenize(text):
  doc = nlp(text)

  # Analyze syntax
  nounChunks=[chunk.text for chunk in doc.noun_chunks]
  verbChunks=[token.lemma_ for token in doc if token.pos_ == "VERB"]
  print("Noun phrases:", nounChunks)
  print("Verbs:", verbChunks)

  # Find named entities, phrases and concepts
  for entity in doc.ents:
    print(entity.text, entity.label_)