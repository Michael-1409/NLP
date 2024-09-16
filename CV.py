import nltk
import spacy
from transformers import pipeline

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Initialize SpaCy model for advanced NLP tasks
nlp = spacy.load('en_core_web_sm')

# Initialize transformers pipeline for sentiment analysis
sentiment_analyzer = pipeline('sentiment-analysis')

# 1. Tokenization
def tokenize_text(text):
    word_tokens = word_tokenize(text)
    sentence_tokens = sent_tokenize(text)
    return word_tokens, sentence_tokens

# 2. Lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    pos_tags = pos_tag(word_tokens)
    
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return lemmatized_words

# 3. Named Entity Recognition (NER) using SpaCy
def named_entity_recognition(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# 4. Sentiment Analysis using Hugging Face Transformers
def analyze_sentiment(text):
    sentiment = sentiment_analyzer(text)
    return sentiment

# Example usage
if __name__ == "__main__":
    text = "Apple is looking at buying U.K. startup for $1 billion. The weather today is great!"

    # Tokenization
    words, sentences = tokenize_text(text)
    print("Word Tokens:", words)
    print("Sentence Tokens:", sentences)
    
    # Lemmatization
    lemmatized_words = lemmatize_text(text)
    print("Lemmatized Words:", lemmatized_words)
    
    # Named Entity Recognition (NER)
    entities = named_entity_recognition(text)
    print("Named Entities:", entities)
    
    # Sentiment Analysis
    sentiment = analyze_sentiment(text)
    print("Sentiment Analysis:", sentiment)
