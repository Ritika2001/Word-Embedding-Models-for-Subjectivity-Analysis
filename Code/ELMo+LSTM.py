
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Lambda, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.engine import Layer
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Dataset/complete10000.csv')
df.head(5)
df.count()

df.drop_duplicates(subset ="text", keep = 'first', inplace = True) 
df.count()

x = df['text'].apply(str).values
y = df['polarity'].values
SEED = 2000

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=SEED)

print ("Train set has total {0} entries with {1:.2f}% objective, {2:.2f}% subjective".format(len(x_train),(len(x_train[y_train == 0]) / (len(x_train)*1.))*100,(len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print ("Test set has total {0} entries with {1:.2f}% objective, {2:.2f}% subjective".format(len(x_test),(len(x_test[y_test == 0]) / (len(x_test)*1.))*100,(len(x_test[y_test == 1]) / (len(x_test)*1.))*100))

# Reduce TensorFlow logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

# Instantiate the elmo model
elmo_module = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)

# Initialize session
sess = tf.Session()
K.set_session(sess)

K.set_learning_phase(1)

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# embed elmo method
def ElmoEmbeddingLayer(x):
    embeddings = elmo_module(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]
    return embeddings

elmo_dim=1024
elmo_input = Input(shape=(None,), dtype=tf.string)
elmo_embedding = Lambda(ElmoEmbeddingLayer, output_shape=(None,elmo_dim))(elmo_input)
x = LSTM(units=70)(elmo_embedding)
x = Dropout(0.5)(x)
x = Dense(1)(x)
x = Activation('relu')(x)
model = Model(inputs=[elmo_input], outputs=x)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, to_file="ELMo.png", show_shapes=True)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('ELMO_LSTM.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, 
                    epochs=200,
                    callbacks=[es, mc],
                    validation_data=(x_test, y_test))



import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model = tf.keras.models.load_model('ELMO_LSTM.h5')


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
plot_confusion_matrix(cm=cm, normalize = False, target_names = classes, title= "Confusion Matrix (ELMO)")


plot_confusion_matrix(cm=cm, normalize = True, target_names = classes, title= "Confusion Matrix (ELMO)")


from sklearn.metrics import classification_report
print("Classification Report: \n",classification_report(y_test, predictions))




