"""

Author: Sang Yuchen
Date:2020/05/31
Desc: mlp for text classification adversarial examples discriminating

"""
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from LSTM_VAE.lstm_vae import get_data_with_EOS_token, rnn_vae, generate_adversarial_examples_pairwise, units_size, z_dim, \
    get_data_without_token

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#超参数设置
vocab_size=10000
seq_length=80
buffer_length=1000
embedding_dim=100
node_num=(embedding_dim * embedding_dim + 1) *2/3     #MLP隐藏层节点数

batch_size=32
epochs=10
learning_rate=0.001
n_samples = 4000
total_samples=12500

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
train_data_with_START_token = keras.preprocessing.sequence.pad_sequences(train_data,
           value=word_index['<PAD>'], padding='post', truncating='post', maxlen=seq_length+1)
test_data_with_START_token = keras.preprocessing.sequence.pad_sequences(test_data,
           value=word_index['<PAD>'], padding='post', truncating='post', maxlen=seq_length + 1)


db_train=tf.data.Dataset.from_tensor_slices((train_data_with_START_token,train_labels))

ds_label_0 = tf.data.Dataset.from_tensor_slices([data for data, label in db_train.as_numpy_iterator()
                                           if label == 0])
ds_label_1 = tf.data.Dataset.from_tensor_slices([data for data, label in db_train.as_numpy_iterator()
                                           if label == 1])

ds_label_0 = ds_label_0.shuffle(buffer_size=buffer_length).batch(batch_size=batch_size,drop_remainder=True)
ds_label_1 = ds_label_1.shuffle(buffer_size=buffer_length).batch(batch_size=batch_size,drop_remainder=True)
ds = [ds_label_0, ds_label_1]

class mlp(keras.Model):
    def __init__(self, vocab_size, embedding_dim, node_num):
        super(mlp, self).__init__()

        self.vocab_size = vocab_size  # 词汇表个数
        self.embedding_dim = embedding_dim  # 单词嵌入表示维度
        self.hidden_units_num = node_num  # 隐藏层节点个数

        self.dense = keras.layers.Dense(self.hidden_units_num, activation='relu')
        self.dense_output = keras.layers.Dense(1)

    def discriminator(self, x):
        """
            :param x: input,[batch_size,seq_size]
            :return: discrimination_result
        """
        x_flaten = tf.reshape(x, [batch_size, -1])  #使用MLP分类，首先将每一个句子展平
        #x_flaten: [batch_size, seq_size * vocab_size]
        hidden_state = self.dense(x_flaten)
        # hidden_state.shape: [batch_size, hidden_units_num]
        output = self.dense_output(hidden_state)
        # output.shape: [batch_size, 1]
        return output

    def call(self, input):
        x = gumbel_softmax(input)               #类似one-hot的字预测结果
        result = self.discriminator(x)
        return x, result


def gumbel_softmax(preds, tau=1):
    epsilon_array = tf.random.uniform([preds.shape[0], seq_length+1, embedding_dim], minval=0, maxval=1)
    G_array = - tf.math.log(- tf.math.log(epsilon_array))
    v_array = tf.add(preds, G_array)
    x = tf.nn.softmax(tf.divide(v_array, tau))
    return x


def MLP_train(D_model_list, G):
    total_loss = []
    D_0 = D_model_list[0]
    D_1 = D_model_list[1]

    for epoch in range(epochs):
        start = time.time()

        step = 0
        for input_label_0, input_label_1 in tf.data.Dataset.zip((ds_label_0, ds_label_1)):

            _, generated_inputs_label_0,_,_ = generate_adversarial_examples_pairwise(G,
                          input_label_0, labels=tf.constant([0] * batch_size))
            _, generated_inputs_label_1,_,_ = generate_adversarial_examples_pairwise(G,
                          input_label_1, labels=tf.constant([1] * batch_size))
            #shape:[32, 81, 10000]

            E = G.embedding(tf.constant(list(range(vocab_size))))
            emb_0 = G.embedding(input_label_0)
            emb_1 = G.embedding(input_label_1)
            emb_gen_0 = tf.matmul(generated_inputs_label_0, E)
            emb_gen_1 = tf.matmul(generated_inputs_label_1, E)
            emb_list = [emb_0, emb_1]
            emb_gen_list = [emb_gen_0, emb_gen_1]

            with tf.GradientTape() as tape:

                total_disc_loss = tf.Variable(tf.constant(0.0))

                for i, D in enumerate(D_model_list):
                    _, logits_real = D(emb_list[i])
                    _, logits_fake = D(emb_gen_list[i])
                    D_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.constant([[1.0]] * batch_size),
                            logits=logits_real)
                    D_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.constant([[0.0]] * batch_size),
                            logits=logits_fake)

                    total_disc_loss = total_disc_loss + (D_real_loss + D_fake_loss)

                total_disc_loss = tf.reduce_sum(total_disc_loss) / (2 * batch_size)
                print(step, "total_disc_loss:", total_disc_loss.numpy())

            gradients = tape.gradient(-total_disc_loss, D_0.trainable_variables+D_1.trainable_variables)
            optimizer = keras.optimizers.Adam(learning_rate)
            optimizer.apply_gradients(zip(gradients, D_0.trainable_variables+D_1.trainable_variables))

            step += 1

            if step % 100 == 0:
                total_loss.append(-total_disc_loss.numpy())
                print('Epoch: {}/{}'.format(epoch, epochs),
                      'Batch--> {}'.format(step),
                       'total_disc_loss--> {}'.format(-total_disc_loss),
                       'Total time of this epoch is {}'.format(time.time() - start))

    D_0_model_name = './weights/mlp_0.ckpt'
    D_0.save_weights(D_0_model_name)

    D_1_model_name = './weights/mlp_1.ckpt'
    D_1.save_weights(D_1_model_name)

    print('plot the loss')
    plt.plot(total_loss)
    plt.xlabel('batch/100')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    D_0 = mlp(vocab_size, embedding_dim, node_num)
    D_1 = mlp(vocab_size, embedding_dim, node_num)
    G = rnn_vae(vocab_size, embedding_dim, units_size, z_dim)
    model_name = "./weights/rnn_vae.ckpt"
    G.load_weights(model_name)

    optimizer_G = keras.optimizers.Adam(learning_rate)
    D_model_list = [D_0, D_1]
    MLP_train(D_model_list, G)

    #train()
    # """
    # model = mlp(vocab_size, embedding_dim, node_num)
    # model_name = "./weights/mlp.ckpt"
    # model.load_weights(model_name)

    # x, y = next(iter(db_train))
    # inputs = get_data_without_token(x)
    # print(x)


    # x_rec =model(x,)
    # for i in x_rec:
    #     print(decode_review(i))
    # # """
