import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import preprocess

nltk.download('punkt')
nltk.download('wordnet')

if __name__ == "__main__":
    words, classes, documents = preprocess.preprocess_data('data/intents.json')
    train_x, train_y = preprocess.create_training_data(words, classes, documents)

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(np.array(train_x.tolist(), dtype=float), np.array(train_y.tolist(), dtype=float), epochs=200, batch_size=5, verbose=1)

    model.save('chatbot_model.h5')
