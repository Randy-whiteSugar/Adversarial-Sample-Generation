"""

Author: Fang Chen
Date:2020/05/28
Desc: load imdb dataset from keras and do some preprocessing

"""

from tensorflow import keras
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#print('tf.__version__:',tf.__version__)

def decode_review(review,index_word):
    """

    :param review: 一条评论，是一个整数序列
    :return: 一条评论，是一个英文单词序列
    """
    return ' '.join([index_word.get(word,'?')for word in review])

def get_data_with_START_token(vocab_size=10000,seq_length=80):
    """

    :param vocab_size:
    :return:
    data_with_START_token : decoder input data
    shape: sample_num,seq_length+1
    """

    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

    word_index=imdb.get_word_index()
    word_index={k:(v+3) for k,v in word_index.items()}

    word_index['<PAD>']=0
    word_index['<START>']=1
    word_index['<UNK>']=2
    word_index['<EOS>']=3
    index_word=dict([(value,key)for(key,value)in word_index.items()])

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

    return (train_data_with_START_token, train_labels),(test_data_with_START_token, test_labels),(word_index,index_word)


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
        if seq[-1] != 0:
            data_with_EOS_token.append(list(seq) + [3])
        else:
            i = -1
            while seq[i - 1] == 0:
                i -= 1
            data_with_EOS_token.append(list(seq[0:i]) + [3] + [0 for j in range(-1 * i)])
    data_with_EOS_token = np.array(data_with_EOS_token)

    return data_with_EOS_token

if __name__ == '__main__':
    (x,y),(_,_),(_,index_word)=get_data_with_START_token()
    print(type(x[0]))
    print(x.shape)
    print(y.shape)
    print("train_data[0]:",decode_review(x[0],index_word))
