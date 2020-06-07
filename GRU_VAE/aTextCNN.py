"""

Author: Fang Chen
Date:2020/06/02
Desc: TextCNN for classifier

"""

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

vocab_size=10000#词库大小
seq_length=300#句子最大长度
vocab_dim=100#词的emedding维度
num_classes=2#分类类别

(train_x, train_y), (test_x, test_y)=tf.keras.datasets.imdb.load_data(num_words=vocab_size)
train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, value=0, padding='post',maxlen=seq_length)
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x,value=0, padding='post', maxlen=seq_length)

model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, vocab_dim))
model.add(layers.Conv1D(filters=256,kernel_size=2,kernel_initializer='he_normal',
                        strides=1,padding='VALID',activation='relu',name='conv'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dropout(rate=0.5,name='dropout'))
model.add(layers.Dense(num_classes,activation='softmax'))
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
history=model.fit(train_x,train_y,epochs=8,batch_size=128,verbose=1,validation_split=0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valiation'], loc='upper left')
plt.show()



