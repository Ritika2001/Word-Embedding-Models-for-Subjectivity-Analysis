import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from clr_callback import CyclicLR



df = pd.read_csv("complete10000.csv")
df = df.reset_index(drop=True)
df.head(5)


class Preprocess:
  DATA_COLUMN = "text"
  LABEL_COLUMN = "polarity"

  def __init__(self, df, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes
    df_x, df_y = self._prepare(df)
    SEED=2000
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=.1, random_state=SEED)

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [train_x, test_x])
    self.train_y=train_y
    self.test_y=test_y

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[Preprocess.DATA_COLUMN], row[Preprocess.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)


bert_model_name="uncased_L-12_H-768_A-12"

bert_ckpt_dir = bert_model_name
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

classes = df.polarity.unique().tolist()

data = Preprocess(df, tokenizer, classes, max_seq_len=128)

max_seq_len = data.max_seq_len

def create_model(max_seq_len, bert_ckpt_file):

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  x = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(2)(x)
  out = tf.keras.layers.Activation('relu')(x)

  model = keras.Model(inputs=input_ids, outputs=out)
  model.build(input_shape=(None, max_seq_len))

  load_stock_weights(bert, bert_ckpt_file)
        
  return model

model = create_model(data.max_seq_len, bert_ckpt_file)

model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
clr = CyclicLR(mode="triangular", base_lr=1e-7, max_lr=1e-2, step_size= 8 * (data.train_x.shape[0] // 8))

history = model.fit(
  x=data.train_x, 
  y=data.train_y,
  shuffle=True,
  epochs=200,
  callbacks = [es,mc,clr],
  validation_data=(data.test_x, data.test_y)
)

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('ELMo+10epochs+Accuracy.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('ELMo+10epochs+Loss.png')
plt.show()

predictions = model.predict([data.test_x])

pred = np.zeros([1000,1], dtype=int)


for i in range(0, predictions.shape[0]):
    pred[i] = np.where(predictions[i] == np.amax(predictions[i]))

print("Accuracy: ",accuracy_score(data.test_y,pred))
from sklearn.metrics import confusion_matrix

print("Classification Report: \n",classification_report(data.test_y,pred))
conmat = np.array(confusion_matrix(data.test_y, pred, labels=[0,1]))
confusion = pd.DataFrame(conmat, index=['objective', 'subjective'],
                         columns=['predicted_objective','predicted_subjective'])
print (confusion)


# In[ ]:




