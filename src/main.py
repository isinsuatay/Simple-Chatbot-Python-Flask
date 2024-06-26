from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import nltk
import preprocess
import json
import random
import os 

app = Flask(__name__, template_folder='../templates')

model = load_model('chatbot_model.h5')
words, classes, _ = preprocess.preprocess_data('data/intents.json')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [preprocess.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def home():
    print(os.path.abspath(os.path.join(app.root_path, '../templates')))
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    message = request.json['message']
    ints = predict_class(message)
    response = get_response(ints, intents)
    return jsonify({"response": response})

if __name__ == '__main__':
    with open('data/intents.json') as file:
        intents = json.load(file)
    app.run(port=5000, debug=True)