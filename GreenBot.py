from imp import load_module
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intent.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
                break  # Add break statement to exit loop when word is found
    return np.array(bag)


def predict_class(sentence, words):
    bow = bag_of_words(sentence, words)
    padded_bow = np.zeros((1, len(words)))
    padded_bow[0, :min(bow.shape[0], len(words))] = bow[:min(bow.shape[0], len(words))]
    # Set verbose to 0 to hide the progress bar output
    res = model.predict(padded_bow, verbose=0)[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list




def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = "Sorry, I did not understand what you said."
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
    

    
print("Hello I'm a Mental Health Chatbot here to support you.")

while True:
    message = input("Type your Message: ")
    ints = predict_class(message, words)
    print(ints)
    res = get_response(ints, intents)
    print("Greenbot: ", res)

