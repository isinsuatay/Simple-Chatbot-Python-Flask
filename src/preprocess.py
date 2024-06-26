import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import random

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_data(intents_file):
    with open(intents_file) as file:
        data = json.load(file)

    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',']

    for intent in data['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(set(words))

    classes = sorted(set(classes))

    return words, classes, documents

def create_training_data(words, classes, documents):
    training = []
    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = [0] * len(classes)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)

    train_x = np.array([sample[0] for sample in training], dtype=object)
    train_y = np.array([sample[1] for sample in training], dtype=object)

    return train_x, train_y