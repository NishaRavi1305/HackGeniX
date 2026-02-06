import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello world")
print("spaCy model loaded and tested OK")
