import warnings
warnings.filterwarnings('ignore')

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Embedding, LSTM
from keras.engine import Input
from keras.preprocessing import text,sequence
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model


df = pd.read_csv('./Dataset/complete10000.csv')
df.count()


df.drop_duplicates(subset ="text", keep = 'first', inplace = True) 
df.count()

x = df['text'].apply(str).values
y = df['polarity'].values
SEED = 2000

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=SEED)

print ("Train set has total {0} entries with {1:.2f}% objective, {2:.2f}% subjective".format(len(x_train),(len(x_train[y_train == 0]) / (len(x_train)*1.))*100,(len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print ("Test set has total {0} entries with {1:.2f}% objective, {2:.2f}% subjective".format(len(x_test),(len(x_test[y_test == 0]) / (len(x_test)*1.))*100,(len(x_test[y_test == 1]) / (len(x_test)*1.))*100))


tk = text.Tokenizer(num_words=200, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True, split=" ")
tk.fit_on_texts(x_train)
sequences_train = tk.texts_to_sequences(x_train)
sequences_test = tk.texts_to_sequences(x_test)
word_index = tk.word_index
print('Found %s unique tokens.' % len(word_index))


x_train = pad_sequences(sequences_train, maxlen=200)
x_test = pad_sequences(sequences_test , maxlen=200)

w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
print('Google News w2v loaded')


w2v_matrix = np.zeros((len(word_index) + 1, 300))

for word,i in word_index.items():
    if word in w2v.vocab:
        w2v_matrix[i] = w2v[word]

model = Sequential()
model.add(Embedding(len(word_index)+1, w2v_matrix.shape[1], weights=[w2v_matrix], input_length=200))
model.add(LSTM(units=70))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
plot_model(model, to_file="Word2Vec300.png", show_shapes=True)

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('Word2Vec_LSTM.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, 
                    epochs=200,
                    callbacks=[es, mc],
                    validation_data=(x_test, y_test))

import tensorflow as tf

model = tf.keras.models.load_model('Word2Vec_LSTM.h5')


loss, accuracy = model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1)
print('Model Loss: {:0.4f} | Model Accuracy: {:.4f}'.format(loss, accuracy))

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Purples')

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('GroundTruths')
    plt.xlabel('Predictions \n Model Accuracy={:0.4f}% | Model Error={:0.4f}%'.format(accuracy*100, misclass*100))
    plt.show()

predictions = model.predict(x=x_test, batch_size=32)

predictions = [0 if i<0.5 else 1 for i in predictions]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
classes=['objective','subjective']
plot_confusion_matrix(cm=cm, normalize = False, target_names = classes, title= "Confusion Matrix (Word2Vec)")

plot_confusion_matrix(cm=cm, normalize = True, target_names = classes, title= "Confusion Matrix (Glove)")


from sklearn.metrics import classification_report
print("Classification Report: \n",classification_report(y_test, predictions))




