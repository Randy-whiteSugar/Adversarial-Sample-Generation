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
import imdb_data
from TextCNN.TextCNN import get_textCNN_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#print('tf.__version__:',tf.__version__)

#超参数设置
vocab_size=12000
seq_length=100
buffer_length=1000
embedding_dim=100
z_dim=32
units_size=64
batch_size=32
epochs=10
learning_rate=0.001

(train_data_with_START_token, train_labels),(test_data_with_START_token, test_labels),(word_index,index_word)=imdb_data.get_data_with_START_token(vocab_size,seq_length)

db_train=tf.data.Dataset.from_tensor_slices((train_data_with_START_token,train_labels))
db_train=db_train.shuffle(buffer_size=buffer_length).batch(batch_size=batch_size,drop_remainder=True)
db_test=tf.data.Dataset.from_tensor_slices((test_data_with_START_token,test_labels))
db_test=db_test.batch(batch_size=batch_size,drop_remainder=True)

class rnn_vae(keras.Model):

    def __init__(self,nvocab,nemd,rnn_units_num,latent_dim):
        super(rnn_vae,self).__init__()

        self.vocab_size=nvocab       #词汇表个数
        self.embedding_dim=nemd      #单词嵌入表示维度
        self.units_size=rnn_units_num#循环体个数
        self.z_dim=latent_dim        #因变量空间维度


        # encoder 层
        self.embedding=keras.layers.Embedding(self.vocab_size,self.embedding_dim,mask_zero=True)#嵌入层采用了mask处理变长输入
        self.recurrent_encoder=keras.layers.GRU(self.units_size)
        #self.lstm_encode = keras.layers.LSTM(self.units_size)
        self.dense_mu = keras.layers.Dense(self.z_dim, activation='relu')
        self.dense_var = keras.layers.Dense(self.z_dim, activation='relu')

        # decoder层，decoder输入的单词嵌入表用和encoder一样的
        self.embedding_label = keras.layers.Embedding(2, 8)
        self.dense_decode=keras.layers.Dense(self.units_size,activation='relu')
        self.recurrent_decoder=keras.layers.GRU(self.units_size,return_sequences=True,return_state=True,dropout=0.6)
        #self.lstm_decode = keras.layers.LSTM(self.units_size, return_sequences=True, return_state=True,dropout=0.4)
        self.dense_output=keras.layers.Dense(self.vocab_size)

    def encoder(self,x):
        """

        :param x: encoder_input,[batch_size,seq_size]
        :return: mu, log_var of latent normal distribution
        """
        x_emb = self.embedding(x)
        # shape: [batch_size,seq_size,embedding_size]
        mask = self.embedding.compute_mask(x)
        h_state = self.recurrent_encoder(x_emb,mask=mask)
        #h_state = self.lstm_encode(x_emb,mask=mask)
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

    def decoder(self,z,x,label,training):
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
        mask = self.embedding.compute_mask(x)
        output, state_h = self.recurrent_decoder(x_emb, initial_state=prev_state,mask=mask,training=training)
        #output, state_h, state_c = self.lstm_decode(x_emb,initial_state=[prev_state,tf.zeros([prev_state.shape[0], self.units_size])],mask=mask,training=training)
        # output shape: [batch_size,seq_size,unit_size]

        logits=self.dense_output(output)
        # shape: [[batch_size,seq_size,vocab_size]]
        preds = tf.nn.softmax(logits)

       # return logits,preds,(state_h,state_c)
        return logits,preds,state_h

    def call(self,encoder_input,decoder_input,label,training=True):
        mu,log_var=self.encoder(encoder_input)
        z=self.reparameterize(mu,log_var)
        logits, preds, state=self.decoder(z,decoder_input,label,training)
        return logits,preds,state,mu,log_var

@tf.function
def train_func(inputs,decoder_inputs, targets, model, labels, loss_func, optimizer,a):

    with tf.GradientTape() as tape:
        logits, _, state,mu,log_var = model(inputs,decoder_inputs, labels,training=True)

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

        a=1.
        start = time.time()

        for (step,(decoder_inputs,labels)) in enumerate(db_train):
            if epoch <3:
                a=0
            elif epoch<10:
                step_all=(781.*int(epoch-3)+int(step))/273.35-10.
                a=1 / (1 + np.exp(-(step_all)))

            encoder_inputs=imdb_data.get_data_without_token(decoder_inputs)
            targets=imdb_data.get_data_with_EOS_token(encoder_inputs)

            with tf.GradientTape() as tape:
                logits, _, state, mu, log_var = model(encoder_inputs, decoder_inputs, labels, training=True)

                rec_loss = rec_loss_func(targets, logits)
                rec_loss = tf.reduce_sum(rec_loss) / encoder_inputs.shape[0]

                kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
                kl_div = tf.reduce_sum(kl_div) / encoder_inputs.shape[0]

                loss = rec_loss + a * kl_div

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


            if step % 100 == 0:
                total_loss.append(loss.numpy())
                print('Epoch: {}/{}'.format(epoch, epochs),
                      'Batch--> {}'.format(step),
                      'Loss--> {}'.format(loss.numpy()),
                      'rec_loss-->{}'.format(rec_loss.numpy()),
                      'kl_div-->{}'.format((kl_div.numpy())),
                      'a-->{}'.format(a))

        print('Total time of this epoch is {}'.format(time.time() - start))


    model_name = "./weights/rnn_vae_gru_3.ckpt"
    model.save_weights(model_name)
    model.summary()
    print('plot the loss')
    plt.plot(total_loss)
    plt.xlabel('batch/100')
    plt.ylabel('loss')
    plt.show()




def generate_adversarial_examples_pairwise(model,decoder_inputs, labels):
    encoder_inputs = imdb_data.get_data_without_token(decoder_inputs)
    _,preds,_,mu, log_var=model(encoder_inputs,decoder_inputs,labels)
    generated_examples=np.argmax(preds.numpy(),axis=2)
    return generated_examples,mu, log_var

def generate_one_adversarial_example_from_scatch(model,labels):
    z=tf.random_normal([1,model.z_dim], mean=0.0, stddev=1.0)
    label_emb=model.embedding_label(labels)
    prev_state = model.dense_decode(tf.concat([z, label_emb], axis=1))
    x_start=tf.convert_to_tensor([[ word_index['<START>'] ]],dtype=tf.float32)
    x_emb = model.embedding(x_start)
    output, state_h, state_c = model.lstm_decode(x_emb, [prev_state, tf.zeros([prev_state.shape[0], model.units_size])])
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
    generated_words.append(word_index['<EOS>'])
    return generated_words



if __name__ == '__main__':
    #train()

    model = rnn_vae(vocab_size, embedding_dim, units_size, z_dim)
    model_name = "./weights/rnn_vae_gru_3.ckpt"
    model.load_weights(model_name)

    count = 0
    total_time = 0
    total_Ture_predict = 0

    textCNN = get_textCNN_model()

    checkpoint_save_path = "D:/Pycharm/tensorlayer/AdversialSampleGen/TextCNN/checkpoint/text_classifier.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        textCNN.load_weights(checkpoint_save_path)
    else:
        print("目标模型未训练，请先运行TextCNN.py")

    for x,y in db_train:
        start_time = time.time()

        x_rec, _, _ = generate_adversarial_examples_pairwise(model,x,y)

        total_time += (time.time() - start_time)

        for i in x_rec:
            print(imdb_data.decode_review(i, index_word))
        count += batch_size

        # 计算攻击成功率
        input_label_0_without_token = imdb_data.get_data_without_token(x_rec)
        pred_result = tf.argmax(textCNN.predict(input_label_0_without_token), axis=1)

        True_count = 0
        for i, j in zip(y, pred_result):
            if i.numpy() == j.numpy():
                True_count += 1
        total_Ture_predict += True_count

        if count > 500:
            print("当前已生成样本", count, "所用时间", total_time, "正确率", total_Ture_predict/count)
            break






