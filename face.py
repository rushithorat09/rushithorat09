import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Preprocess the data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Add documents to the corpus
        documents.append((word, intent['tag']))
        # Add to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lowercase each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))
# Print statistics
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes to files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create the training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training_item = [np.array(bag), np.array(output_row)]
    try:
        training.append(training_item)
    except Exception as e:
        print("Error occurred:", e)
        print("Training item:", training_item)

# Shuffle the features
random.shuffle(training)
training = np.array(training)

# Create training and testing lists: X - patterns, Y - intents
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))
# Deep neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train and save the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Model is created")
