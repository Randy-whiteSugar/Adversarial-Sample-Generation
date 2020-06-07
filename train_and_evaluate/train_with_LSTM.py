import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from LSTM_VAE.lstm_vae import get_data_with_EOS_token, get_data_without_token, rnn_vae, \
    generate_adversarial_examples_pairwise
from MLP.mlp import mlp, node_num, ds_label_0, ds_label_1
from TextCNN.TextCNN import get_textCNN_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_with_LSTM():
    #超参数设置
    vocab_size=10000
    embedding_dim=100
    z_dim=10
    units_size=64

    learning_rate=0.001
    epochs = 10
    batch_size = 32

    #模型
    D_0 = mlp(vocab_size, embedding_dim, node_num)
    D_1 = mlp(vocab_size, embedding_dim, node_num)
    optimizer = keras.optimizers.Adam(learning_rate)
    D_model_list = [D_0, D_1]

    G = rnn_vae(vocab_size, embedding_dim, units_size, z_dim)
    optimizer_G = keras.optimizers.Adam(learning_rate)

    checkpoint_save_path = "./weights/D_0.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        # print('-------------load the model-----------------')
        D_0.load_weights(checkpoint_save_path)
        D_1.load_weights("./weights/D_1.ckpt")
        G.load_weights("./weights/G.ckpt")
        print("已加载模型参数")
    else:
        print("生成器和判别器都未训练")
        return

    total_loss = []
    D_0 = D_model_list[0]
    D_1 = D_model_list[1]

    for epoch in range(epochs):
        start = time.time()

        step = 0
        for input_label_0, input_label_1 in tf.data.Dataset.zip((ds_label_0, ds_label_1)):

            a = 1
            with tf.GradientTape(persistent=True) as tape:
                #VAE产生样本
                _, generated_inputs_label_0, mu, log_var = generate_adversarial_examples_pairwise(
                        G, input_label_0, labels=tf.constant([0] * batch_size))
                _, generated_inputs_label_1, mu, log_var = generate_adversarial_examples_pairwise(
                        G, input_label_1, labels=tf.constant([1] * batch_size))

                input_label_0_without_token = get_data_without_token(input_label_0)
                input_label_1_without_token = get_data_without_token(input_label_1)

                input_label_0_with_EOS = get_data_with_EOS_token(input_label_0_without_token)
                input_label_1_with_EOS = get_data_with_EOS_token(input_label_1_without_token)

                #计算Disc_loss
                disc_loss_0 = tf.keras.losses.sparse_categorical_crossentropy(input_label_0_with_EOS,
                                                                              generated_inputs_label_0)
                disc_loss_1 = tf.keras.losses.sparse_categorical_crossentropy(input_label_1_with_EOS,
                                                                              generated_inputs_label_1)

                total_disc_loss = tf.reduce_sum(disc_loss_0 + disc_loss_1) / (2*batch_size)

                #计算VAE_loss
                targets = tf.concat([input_label_0_with_EOS, input_label_1_with_EOS], axis=0)

                generated_samples = tf.concat([generated_inputs_label_0, generated_inputs_label_1], axis=0)

                rec_loss_func = keras.losses.SparseCategoricalCrossentropy()
                rec_loss = rec_loss_func(targets, generated_samples)
                rec_loss = tf.reduce_sum(rec_loss) / (2 * batch_size)

                kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
                kl_div = tf.reduce_sum(kl_div) / (2 * batch_size)

                vae_loss = rec_loss + a * kl_div

                #计算adv_loss
                textCNN = get_textCNN_model()

                checkpoint_save_path = "D:/Pycharm/tensorlayer/AdversialSampleGen/TextCNN/checkpoint/text_classifier.ckpt"
                if os.path.exists(checkpoint_save_path + '.index'):
                    textCNN.load_weights(checkpoint_save_path)
                else:
                    print("目标模型未训练，请先运行TextCNN.py")
                    return

                inputs_without_token = tf.concat([input_label_0_without_token,input_label_1_without_token], axis=0)
                real_labels = tf.one_hot(tf.constant([0]*batch_size +[1]*batch_size),depth=2)
                pred_result = textCNN.predict(inputs_without_token)
                bce = tf.keras.losses.BinaryCrossentropy()
                adv_loss = bce(real_labels, pred_result) / (2 * batch_size)

                joint_loss = total_disc_loss + 100 * vae_loss + 100 * adv_loss

            gradients = tape.gradient(-total_disc_loss, D_0.trainable_variables + D_1.trainable_variables)
            optimizer.apply_gradients(zip(gradients, D_0.trainable_variables + D_1.trainable_variables))

            grad = tape.gradient(joint_loss, G.trainable_weights)
            optimizer_G.apply_gradients(zip(grad, G.trainable_weights))

            step += 1
            if step % 100 == 0:
                total_loss.append(joint_loss.numpy())
                print('Epoch: {}/{}'.format(epoch, epochs),
                      'Batch--> {}'.format(step),
                      'total_loss--> {}'.format(joint_loss),
                      'adv_loss--> {}'.format(adv_loss),
                      'vae_loss--> {}'.format(vae_loss),
                      'total_disc_loss--> {}'.format(total_disc_loss),
                      'Total time of this epoch is {}'.format(time.time() - start))

    D_0_model_name = './weights/D_0_2.ckpt'
    D_0.save_weights(D_0_model_name)

    D_1_model_name = './weights/D_1_2.ckpt'
    D_1.save_weights(D_1_model_name)

    G_model_name = './weights/G_2.ckpt'
    G.save_weights(G_model_name)

    print('plot the loss')
    plt.plot(total_loss)
    plt.xlabel('batch/100')
    plt.ylabel('loss')
    plt.show()



if __name__ == '__main__':
    train_with_LSTM()