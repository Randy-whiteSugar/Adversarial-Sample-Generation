"""

Author: Fang Chen
Date:2020/05/28
Desc: rnn_vae for text classification adversarial examples generating

"""
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#print('tf.__version__:',tf.__version__)

#超参数设置
vocab_size=10000
seq_length=80
buffer_length=1000
embedding_dim=100
z_dim=10
units_size=64
batch_size=32
epochs=10
learning_rate=0.001

#数据处理
imdb=keras.datasets.imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=vocab_size)

word_index=imdb.get_word_index()
word_index={k:(v+3) for k,v in word_index.items()}

word_index['<PAD>']=0
word_index['<START>']=1
word_index['<UNK>']=2
word_index['<EOS>']=3

index_word=dict([(value,key)for(key,value)in word_index.items()])

def decode_review(review):
    """

    :param review: 一条评论，是一个整数序列
    :return: 一条评论，是一个英文单词序列
    """
    return ' '.join([index_word.get(word,'<UNK>')for word in review])

#data_with_START_token :
# shape: sample_num,seq_length+1
# Desc: decoder input data
train_data_with_START_token=keras.preprocessing.sequence.pad_sequences(train_data,
                                                                   value=word_index['<PAD>'],
                                                                   padding='post',
                                                                   truncating='post',
                                                                   maxlen=seq_length+1)
test_data_with_START_token=keras.preprocessing.sequence.pad_sequences(test_data,
                                                                  value=word_index['<PAD>'],
                                                                  padding='post',
                                                                  truncating='post',
                                                                  maxlen=seq_length+1)
db_train=tf.data.Dataset.from_tensor_slices((train_data_with_START_token,train_labels))
db_train=db_train.shuffle(buffer_size=buffer_length).batch(batch_size=batch_size,drop_remainder=True)
db_test=tf.data.Dataset.from_tensor_slices((test_data_with_START_token,test_labels))
db_test=db_test.batch(batch_size=batch_size,drop_remainder=True)


def get_data_without_token(data_with_START_token):
    """

    :param data_with_START_token: raw_padding_train_data
    :return:
        data_without_token
        shape: sample_num,seq_length
        Desc: encoder input data

    """
    data_without_token = []
    for seq in data_with_START_token:
        data_without_token.append(seq[1:])
    data_without_token = np.array(data_without_token)

    return data_without_token

def get_data_with_EOS_token(data_without_token):
    """

    :param data_without_token:
    :return:
        data_with_EOS_token
        shape: sample_num,seq_length+1
        Desc: decoder output baseline
    """
    data_with_EOS_token = []
    for seq in data_without_token:
        if seq[-1] != word_index['<PAD>']:
            data_with_EOS_token.append(list(seq) + [word_index['<EOS>']])
        else:
            i = -1
            while seq[i - 1] == word_index['<PAD>']:
                i -= 1
            data_with_EOS_token.append(list(seq[0:i]) + [word_index['<EOS>']] + [word_index['<PAD>'] for j in range(-1 * i)])
    data_with_EOS_token = np.array(data_with_EOS_token)

    return data_with_EOS_token



class rnn_vae(keras.Model):
    def __init__(self,nvocab,nemd,rnn_units_num,latent_dim):
        super(rnn_vae,self).__init__()

        self.vocab_size=nvocab       #词汇表个数
        self.embedding_dim=nemd      #单词嵌入表示维度
        self.units_size=rnn_units_num#循环体个数
        self.z_dim=latent_dim        #因变量空间维度


        # encoder 层
        self.embedding=keras.layers.Embedding(self.vocab_size,self.embedding_dim,mask_zero=True)#嵌入层采用了mask处理变长输入
        self.lstm_encode = tf.keras.layers.LSTM(self.units_size)
        self.dense_mu = keras.layers.Dense(self.z_dim, activation='relu')
        self.dense_var = keras.layers.Dense(self.z_dim, activation='relu')

        # decoder层，decoder输入的单词嵌入表用和encoder一样的
        self.embedding_label = keras.layers.Embedding(2, self.z_dim)
        self.dense_decode=keras.layers.Dense(self.units_size,activation='relu')
        self.lstm_decode = tf.keras.layers.LSTM(self.units_size, return_sequences=True, return_state=True)
        self.dense_output=keras.layers.Dense(self.vocab_size)

    def encoder(self,x):
        """

        :param x: encoder_input,[batch_size,seq_size]
        :return: mu, log_var of latent normal distribution
        """
        x_emb = self.embedding(x)
        # shape: [batch_size,seq_size,embedding_size]
        h_state = self.lstm_encode(x_emb)
        # shape: [batch_size,unit_size]
        mu=self.dense_mu(h_state)
        # shape: [batch_size,z_dim]
        log_var=self.dense_var(h_state)
        # shape: [batch_size,z_dim]
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        reparameterize trick
        :param mu:
        :param log_var:
        :return: sampled latent vector
        """
        epsilon = tf.random.normal(log_var.shape)
        std = tf.exp(log_var) ** 0.5
        z = mu + std * epsilon
        # shape: [batch_size,z_dim]
        return z

    def decoder(self,z,x,label):
        """

        :param z: latent vector
        :param x: decoder input
        :param label: input label
        :return:
        """
        # shape: [batch_size,]
        label_emb=self.embedding_label(label)
        # shape: [batch_size,label_embedding_size]
        prev_state=self.dense_decode(tf.concat([z,label_emb],axis=1))
        # shape: [batch_size,unit_size]
        x_emb=self.embedding(x)
        # shape: [batch_size,seq_size,embedding_size]
        output, state_h, state_c = self.lstm_decode(x_emb, [prev_state,prev_state])
        # output shape: [batch_size,seq_size,unit_size]


        logits=self.dense_output(output)
        # shape: [[batch_size,seq_size,vocab_size]]
        preds = tf.nn.softmax(logits)

        return logits,preds,(state_h,state_c)

    def call(self,encoder_input,decoder_input,label):
        mu,log_var=self.encoder(encoder_input)
        z=self.reparameterize(mu,log_var)
        logits, preds, (state_h, state_c)=self.decoder(z,decoder_input,label)
        return logits,preds,(state_h,state_c),mu,log_var

@tf.function
def train_func(inputs,decoder_inputs, targets, model, labels, loss_func, optimizer,a):

    with tf.GradientTape() as tape:
        logits, _, state,mu,log_var = model(inputs,decoder_inputs, labels)

        rec_loss = loss_func(targets, logits)
        rec_loss = tf.reduce_sum(rec_loss) / inputs.shape[0]

        kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
        kl_div = tf.reduce_sum(kl_div) / inputs.shape[0]

        loss=rec_loss+a*kl_div

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss,rec_loss,kl_div

def train():

    model=rnn_vae(vocab_size,embedding_dim,units_size,z_dim)

    optimizer=keras.optimizers.Adam(learning_rate)
    rec_loss_func=keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    total_loss = []
    for epoch in range(epochs):
        start = time.time()
        for (step,(decoder_inputs,labels)) in enumerate(db_train):

            a=1

            encoder_inputs=get_data_without_token(decoder_inputs)
            targets=get_data_with_EOS_token(encoder_inputs)

            loss,rec_loss,kl_div=train_func(encoder_inputs,decoder_inputs,targets,model,labels,rec_loss_func,optimizer,a)

            if step % 100 == 0:
                total_loss.append(loss.numpy())
                print('Epoch: {}/{}'.format(epoch, epochs),
                      'Batch--> {}'.format(step),
                      'Loss--> {}'.format(loss.numpy()),
                      'rec_loss-->{}'.format(rec_loss.numpy()),
                      'kl_div-->{}'.format((kl_div.numpy())))

        print('Total time of this epoch is {}'.format(time.time() - start))

    model.summary()
    model_name = "./weights/rnn_vae_2.ckpt"
    model.save_weights(model_name)
    print('plot the loss')
    plt.plot(total_loss)
    plt.xlabel('batch/100')
    plt.ylabel('loss')
    plt.show()


def generate_adversarial_examples_pairwise(model, decoder_inputs, labels):
    encoder_inputs = get_data_without_token(decoder_inputs)
    _,preds,(_,_), mu, log_var=model(encoder_inputs,decoder_inputs,labels)
    generated_examples=np.argmax(preds.numpy(),axis=2)
    return generated_examples, preds, mu, log_var

def generate_one_adversarial_example_from_scatch(model,labels):
    z=tf.keras.backend.random_normal([1,model.z_dim], mean=0.0, stddev=1.0)
    label_emb=model.embedding_label(labels)
    prev_state = model.dense_decode(tf.concat([z, label_emb], axis=1))
    x_start=tf.convert_to_tensor([[word_index['<START>'] ]],dtype=tf.float32)
    x_emb = model.embedding(x_start)
    output, state_h, state_c = model.lstm_decode(x_emb, [prev_state, prev_state])
    logits = model.dense_output(output)
    preds = tf.nn.softmax(logits)
    generated_words=[]
    generated_word=np.argmax(np.squeeze(preds.numpy()))
    while generated_word!=word_index['<EOS>'] and len(generated_words)<=seq_length:
        generated_words.append(generated_word)
        x_input=tf.convert_to_tensor([[ generated_word ]],dtype=tf.float32)
        x_emb = model.embedding(x_input)
        output, state_h, state_c = model.lstm_decode(x_emb, [state_h, state_c])
        logits = model.dense_output(output)
        preds = tf.nn.softmax(logits)
        generated_word = np.argmax(np.squeeze(preds.numpy()))
    generated_words.append(generated_word)
    return generated_words

if __name__ == '__main__':
    #train()
    # """
    model = rnn_vae(vocab_size, embedding_dim, units_size, z_dim)
    model_name = "./weights/rnn_vae.ckpt"
    model.load_weights(model_name)

    x, y = next(iter(db_train))
    print(y)
    # x_rec=generate_adversarial_examples_pairwise(model,x,y)
    # for i in x_rec:
    #     print(decode_review(i))


    # print(y[0])
    # print(tf.reshape(y[0], [1]))
    # gen_x = generate_one_adversarial_example_from_scatch(model, tf.reshape(y[0], [1]))
    #
    # print(decode_review(gen_x))

    # """




