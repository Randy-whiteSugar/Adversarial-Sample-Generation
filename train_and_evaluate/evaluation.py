import time
import tensorflow as tf
import os
from LSTM_VAE.lstm_vae import rnn_vae, generate_adversarial_examples_pairwise
from MLP.mlp import ds_label_0, ds_label_1, index_word
from TextCNN.TextCNN import get_textCNN_model
from imdb_data import get_data_without_token, decode_review

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def expe():
    #超参数设置
    vocab_size=10000
    embedding_dim=100
    z_dim=10
    units_size=64

    learning_rate=0.001
    epochs = 10
    batch_size = 32

    #模型
    G = rnn_vae(vocab_size, embedding_dim, units_size, z_dim)

    checkpoint_save_path = "./weights/G.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        G.load_weights("./weights/G.ckpt")
        print("已加载生成器参数")
    else:
        print("生成器未训练")
        return

    textCNN = get_textCNN_model()

    checkpoint_save_path = "D:/Pycharm/tensorlayer/AdversialSampleGen/TextCNN/checkpoint/text_classifier.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        textCNN.load_weights(checkpoint_save_path)
    else:
        print("目标模型未训练，请先运行TextCNN.py")
        return

    total_gen_time = 0
    total_Ture_predict = 0
    for epoch in range(epochs):
        start = time.time()
        for input_label_0, input_label_1 in tf.data.Dataset.zip((ds_label_0, ds_label_1)):
            #样本生产
            generated_example_0, generated_inputs_label_0, _, _ = generate_adversarial_examples_pairwise(
                    G, input_label_0, labels=tf.constant([0] * batch_size))
            generated_example_1, generated_inputs_label_1, _, _ = generate_adversarial_examples_pairwise(
                    G, input_label_1, labels=tf.constant([1] * batch_size))

            #时间计算
            total_gen_time += (time.time() - start)

            #样本质量
            with open('genSample.txt', 'a', encoding='utf-8') as f:
                f.write("\n------------label=0的生成样本-----------\n")
                for i in generated_example_0:
                    f.write(decode_review(i, index_word) + '\n')
                f.write("\n------------label=1的生成样本-----------\n")
                for j in generated_example_1:
                    f.write(decode_review(j, index_word) + '\n')

            #计算攻击成功率
            input_label_0_without_token = get_data_without_token(input_label_0)
            input_label_1_without_token = get_data_without_token(input_label_1)
            inputs_without_token = tf.concat([input_label_0_without_token, input_label_1_without_token], axis=0)
            real_labels = tf.constant([0] * batch_size + [1] * batch_size)
            pred_result = tf.argmax(textCNN.predict(inputs_without_token), axis=1)

            True_count = 0
            for i, j in zip(real_labels, pred_result):
                if i.numpy() == j.numpy():
                    True_count += 1
            total_Ture_predict += True_count

    print("总生成用时：" + str(total_gen_time) + '\n' + "平均单样本生成时间：", str(total_gen_time/(batch_size *2* epochs)))
    print("正确数：" + str(total_Ture_predict) + '\n' + "正确率：", str(total_Ture_predict/(batch_size *2* epochs)))


if __name__ == '__main__':
    expe()