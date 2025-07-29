import pandas as pd
import numpy as np
import os
from keras.src.layers import Dense, Activation, Flatten, Dropout, Reshape,Input,Lambda,LSTM, Bidirectional, PReLU
from keras.src.layers import Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling1D, GlobalMaxPooling2D, AveragePooling1D
from keras.src.models import Sequential, Model
from keras.src.utils import to_categorical
from keras.src.optimizers import Adam, SGD
from keras.src.layers import BatchNormalization
from keras.src.regularizers import L2
import keras.src.backend as K
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.src.layers import MultiHeadAttention, LayerNormalization, Add
from keras.src.layers import concatenate
from keras.src.layers import Dense, concatenate, Layer
from keras.src.activations import softmax
from keras.src.legacy.backend import batch_dot,sqrt,cast,int_shape
from keras.src.layers import Dense,Reshape,multiply,Permute
from keras.src.legacy.backend import int_shape
from keras.src.legacy import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv,math
from keras.src.layers import MultiHeadAttention, LayerNormalization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def transformer_block(x, nb_filter, num_heads=4, ff_dim=128, rate=0.2):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=nb_filter)(x, x)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)
    attn_output = Dropout(rate)(attn_output)
    ffn_output = Dense(ff_dim, activation='relu')(attn_output)
    ffn_output = Dense(nb_filter)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)
    return Dropout(rate)(ffn_output)
def conv_factory(x, init_form, nb_filter, filter_size_block, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, filter_size_block,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=L2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def transition(x, init_form, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, kernel_size=1,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=L2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    #x = AveragePooling2D((2, 2),padding='same')(x)
    x = AveragePooling1D(pool_size=2, padding='same')(x)

    return x

def denseblock(x, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    list_feat = [x]
    concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, init_form, growth_rate, filter_size_block, dropout_rate, weight_decay)
        list_feat.append(x)
        #x = concatenate(list_feat, mode='concat', concat_axis=concat_axis)
        x = concatenate(list_feat, axis=-1)
        nb_filter += growth_rate
    return x


import numpy as np
def attention_block(input_tensor):
    """ Attention mechanism in the form of squeeze and excitation block """
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling1D()(input_tensor)
    se = Dense(filters // 16, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, filters))(se)
    return multiply([input_tensor, se])




def multi_head_attention_fusion(x1, x2, x3, num_heads, Daxis):
    # Ensure that x1, x2, x3 have the same shape
    # assert x1.shape[1:] == x2.shape[1:] == x3.shape[1:], "Inputs must have the same shape."

    # Concatenate inputs along the feature dimension

    concatenated_inputs = concatenate([x1, x2, x3], axis=Daxis)

    # Create a MultiHeadAttention layer
    attention_layer = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=concatenated_inputs.shape[-1] // num_heads
    )

    # Apply Multi-Head Attention
    attention_output = attention_layer(query=concatenated_inputs, value=concatenated_inputs, key=concatenated_inputs)

    # Add & Norm
    attention_output = Add()([concatenated_inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)

    return attention_output

def two_head_attention_fusion(x1, x2,  num_heads,axis):
    # Ensure that x1, x2, x3 have the same shape
    # assert x1.shape[1:] == x2.shape[1:] == x3.shape[1:], "Inputs must have the same shape."

    # Concatenate inputs along the feature dimension

    concatenated_inputs = concatenate([x1, x2], axis=axis)

    # Create a MultiHeadAttention layer
    attention_layer = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=concatenated_inputs.shape[-1] // num_heads
    )

    # Apply Multi-Head Attention
    attention_output = attention_layer(query=concatenated_inputs, value=concatenated_inputs, key=concatenated_inputs)

    # Add & Norm
    attention_output = Add()([concatenated_inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)

    return attention_output
def residual_block(x, init_form, nb_filter, filter_size, weight_decay):
    shortcut = x
    x = Conv1D(filters=nb_filter,kernel_size=filter_size,
               kernel_initializer=init_form,
               activation='relu',
               padding='same',
               use_bias=False,
               kernel_regularizer=L2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=nb_filter,kernel_size=filter_size,
               kernel_initializer=init_form,
               activation='relu',
               padding='same',
               use_bias=False,
               kernel_regularizer=L2(weight_decay))(x)
    x = BatchNormalization()(x)
    shortcut = Conv1D(filters=nb_filter,kernel_size=filter_size,
               kernel_initializer=init_form,
               activation='relu',
               padding='same',
               use_bias=False,
               kernel_regularizer=L2(weight_decay))(shortcut)
    x = Add()([x, shortcut])
    return x



def weighted_binary_crossentropy(y_true, y_pred):
    # 计算每个类别的权重
    class_weights = K.sum(y_true) / (K.sum(1 - y_true) + 0.000001)  # 根据样本类别分布自动调整权重
    # 计算加权的二分类交叉熵损失
    loss = K.mean(class_weights * K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss


def mid_loss(y_true, y_pred):
    eps = K.epsilon()  # Small constant to avoid division by zero

    # Convert y_true to one-hot encoded vectors if not already
    if K.ndim(y_true) == 1:
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)  # Assuming binary classification with 2 classes

    # Extract the positive class from y_true and y_pred
    y_true_pos = y_true[:, 1]
    y_pred_pos = y_pred[:, 1]

    # Calculate positive and negative losses
    pos = y_true_pos * y_pred_pos / K.maximum(eps, y_true_pos)
    pos = -K.log(pos + eps)

    neg = (1 - y_true_pos) * y_pred_pos / K.maximum(eps, 1 - y_true_pos)
    neg = K.abs(neg - 1e-2)
    neg = -K.log(1 - neg)

    # Combine positive and negative losses with weights
    return K.mean(0.95 * pos + 0.05 * neg, axis=-1)

def count_csv_files(folder_path):
    """
    统计文件夹中的 CSV 文件数量
    """
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pssm'):
            count += 1
    return count



def read_pssm(file_path,window_size):
    # 读取数据，跳过前两行（假设文件前两行是注释或元数据）
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=3, header=None)

    # 获取所有列的名称（需要根据实际数据调整列名）
    columns = [
        "Pos", "Residue",
        "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
        "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
        "K_mean", "Lambda"
    ]

    # 为 DataFrame 指定列名
    df.columns = columns

    default_column = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y",
                      "V"]
    # 提取 A 到 V 列
    df_sub = df.loc[:, default_column]


    df_sub = df_sub.loc[:,::2]
    # 提取前 41 行
    df_sub = df_sub.head(33)

    df_sub = df_sub.iloc[16-int((window_size-1)/2):17+int((window_size-1)/2),:]

    return df_sub


# 将 DataFrame 转换为 TensorFlow tensor
def df_to_tensor(df):
    # 将 DataFrame 转换为 numpy 数组
    numpy_array = df.to_numpy()

    # 将 numpy 数组转换为 TensorFlow tensor
    tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)  # 选择适当的数据类型

    return tensor


def process_all_pssm_files(folder_path,window_size):
    all_tensors = []

    # 遍历文件夹中的所有 PSSM 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".pssm"):
            file_path = os.path.join(folder_path, filename)

            # 读取并处理 PSSM 文件
            df_extracted = read_pssm(file_path, window_size)

            # 将 DataFrame 转换为 tensor
            tensor = df_to_tensor(df_extracted)

            # 添加到列表中
            all_tensors.append(tensor)

    # 将所有 tensor 合并成一个大的矩阵
    all_tensors_matrix = tf.stack(all_tensors)

    return all_tensors_matrix


def Methys(nb_classes, nb_layers, img_BLdim1, img_BLdim2, img_BLdim3, img_PSSMdim1, img_PSSMdim2, img_PSSMdim3,
           init_form, nb_dense_block,
           growth_rate, filter_size_block1, filter_size_block2, filter_size_block3,
           nb_filter, filter_size_ori,
           dense_number, dropout_rate, dropout_dense, weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    # first input of 33 seq #
    main_input = Input(shape=img_BLdim1)
    main_input_transpose = Permute((2, 1))(main_input)
    # model_input = Input(shape=img_dim)
    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer=init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                # W_regularizer=l2(weight_decay))(main_input)
                kernel_regularizer=L2(weight_decay))(main_input)
    x1_transpose = Conv1D(nb_filter, filter_size_ori,
                          kernel_initializer=init_form,
                          activation='relu',
                          padding='same',
                          use_bias=False,
                          # W_regularizer=l2(weight_decay))(main_input)
                          kernel_regularizer=L2(weight_decay))(main_input_transpose)
    # x1 = transformer_block(main_input, nb_filter)
    # x1_transpose = transformer_block(main_input_transpose,nb_filter)
    # x1 = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x1)
    # x1_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x1_transpose)
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        x1_transpose = denseblock(x1_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        x1_transpose = transition(x1_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                    dropout_rate=dropout_rate,
                    weight_decay=weight_decay)
    x1_transpose = denseblock(x1_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x1 = residual_block(x1, init_form, nb_filter, filter_size_block1, weight_decay)
    x1_transpose = residual_block(x1_transpose, init_form, nb_filter, filter_size_block1, weight_decay)

    # x1 = PReLU()(x1)
    # x1_transpose = PReLU()(x1_transpose)
    x1 = Activation('relu')(x1)
    x1_transpose = Activation('relu')(x1_transpose)

    # x1 = two_head_attention_fusion(x1,x1_transpose,2)

    # second input of 21 seq #
    input2 = Input(shape=img_BLdim2)
    input2_transpose = Permute((2, 1))(input2)
    x2 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer=init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=L2(weight_decay))(input2)
    x2_transpose = Conv1D(nb_filter, filter_size_ori,
                          kernel_initializer=init_form,
                          activation='relu',
                          padding='same',
                          use_bias=False,
                          kernel_regularizer=L2(weight_decay))(input2_transpose)
    # x2 = transformer_block(input2, nb_filter)
    # x2_transpose = transformer_block(input2_transpose,nb_filter)
    # x2 = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x2)
    # x2_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x2_transpose)
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        x2_transpose = denseblock(x2_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x2 = transition(x2, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        x2_transpose = transition(x2_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                    dropout_rate=dropout_rate,
                    weight_decay=weight_decay)
    x2_transpose = denseblock(x2_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x2 = residual_block(x2, init_form, nb_filter, filter_size_block2, weight_decay)
    x2_transpose = residual_block(x2_transpose, init_form, nb_filter, filter_size_block2, weight_decay)

    x2 = Activation('relu')(x2)
    x2_transpose = Activation('relu')(x2_transpose)
    # x2 = PReLU()(x2)
    # x2_transpose = PReLU()(x2_transpose)

    # x2 = two_head_attention_fusion(x2,x2_transpose,2)

    # third input seq of 15 #
    input3 = Input(shape=img_BLdim3)
    input3_transpose = Permute((2, 1))(input3)
    x3 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer=init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=L2(weight_decay))(input3)
    x3_transpose = Conv1D(nb_filter, filter_size_ori,
                          kernel_initializer=init_form,
                          activation='relu',
                          padding='same',
                          use_bias=False,
                          kernel_regularizer=L2(weight_decay))(input3_transpose)
    # x3 = transformer_block(input3, nb_filter)
    # x3_transpose = transformer_block(input3_transpose,nb_filter)
    # x3 = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x3)
    # x3_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x3_transpose)
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        x3_transpose = denseblock(x3_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x3 = transition(x3, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        x3_transpose = transition(x3_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                    dropout_rate=dropout_rate,
                    weight_decay=weight_decay)
    x3_transpose = denseblock(x3_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)
    x3 = residual_block(x3, init_form, nb_filter, filter_size_block3, weight_decay)
    x3_transpose = residual_block(x3_transpose, init_form, nb_filter, filter_size_block3, weight_decay)

    x3 = Activation('relu')(x3)
    x3_transpose = Activation('relu')(x3_transpose)
    # x3 = PReLU()(x3)
    # x3_transpose = PReLU()(x3_transpose)
    # x3 = two_head_attention_fusion(x3, x3_transpose,2)

    # x=concatenate([x1,x2,x3],axis=1)
    # 需要改进!!!

    # x = Flatten()(attention)  有问题
    # x = attention_block(x)
    x = concatenate([x1, x2, x3], axis=1)
    x_transpose = concatenate([x1_transpose, x2_transpose, x3_transpose], axis=2)
    # x = multi_head_attention_fusion(x1, x2, x3, num_heads=6, Daxis=1)
    # x_transpose = multi_head_attention_fusion(x1_transpose,x2_transpose,x3_transpose, num_heads=6, Daxis=2) #

    main_PSSMinput = Input(shape=img_PSSMdim1)
    main_PSSMinput_transpose = Permute((2, 1))(main_PSSMinput)
    # model_input = Input(shape=img_dim)
    # Initial convolution
    x1PSSM = Conv1D(nb_filter, filter_size_ori,
                    kernel_initializer=init_form,
                    activation='relu',
                    padding='same',
                    use_bias=False,
                    # W_regularizer=l2(weight_decay))(main_input)
                    kernel_regularizer=L2(weight_decay))(main_PSSMinput)
    x1PSSM_transpose = Conv1D(nb_filter, filter_size_ori,
                              kernel_initializer=init_form,
                              activation='relu',
                              padding='same',
                              use_bias=False,
                              # W_regularizer=l2(weight_decay))(main_input)
                              kernel_regularizer=L2(weight_decay))(main_PSSMinput_transpose)
    # x1PSSM = transformer_block(main_PSSMinput, nb_filter)
    # x1PSSM_transpose = transformer_block(main_PSSMinput_transpose,nb_filter)
    # x1PSSM = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x1PSSM)
    # x1PSSM_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x1PSSM_transpose)
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x1PSSM = denseblock(x1PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                            dropout_rate=dropout_rate,
                            weight_decay=weight_decay)
        x1PSSM_transpose = denseblock(x1PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate,
                                      filter_size_block1,
                                      dropout_rate=dropout_rate,
                                      weight_decay=weight_decay)
        # add transition
        x1PSSM = transition(x1PSSM, init_form, nb_filter, dropout_rate=dropout_rate,
                            weight_decay=weight_decay)
        x1PSSM_transpose = transition(x1PSSM_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
                                      weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x1PSSM = denseblock(x1PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    x1PSSM_transpose = denseblock(x1PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)

    x1PSSM = residual_block(x1PSSM, init_form, nb_filter, filter_size_block1, weight_decay)
    x1PSSM_transpose = residual_block(x1PSSM_transpose, init_form, nb_filter, filter_size_block1, weight_decay)

    # x1PSSM = PReLU()(x1PSSM)
    # x1PSSM_transpose = PReLU()(x1PSSM_transpose)
    x1PSSM = Activation('relu')(x1PSSM)
    x1PSSM_transpose = Activation('relu')(x1PSSM_transpose)

    # x1 = two_head_attention_fusion(x1,x1_transpose,2)

    # second input of 21 seq #
    inputPSSM2 = Input(shape=img_PSSMdim2)
    inputPSSM2_transpose = Permute((2, 1))(inputPSSM2)
    x2PSSM = Conv1D(nb_filter, filter_size_ori,
                    kernel_initializer=init_form,
                    activation='relu',
                    padding='same',
                    use_bias=False,
                    kernel_regularizer=L2(weight_decay))(inputPSSM2)
    x2PSSM_transpose = Conv1D(nb_filter, filter_size_ori,
                              kernel_initializer=init_form,
                              activation='relu',
                              padding='same',
                              use_bias=False,
                              kernel_regularizer=L2(weight_decay))(inputPSSM2_transpose)
    # x2PSSM = transformer_block(inputPSSM2, nb_filter)
    # x2PSSM_transpose = transformer_block(inputPSSM2_transpose,nb_filter)
    # x2PSSM = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x2PSSM)
    # x2PSSM_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x2PSSM_transpose)
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x2PSSM = denseblock(x2PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                            dropout_rate=dropout_rate,
                            weight_decay=weight_decay)
        x2PSSM_transpose = denseblock(x2PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate,
                                      filter_size_block2,
                                      dropout_rate=dropout_rate,
                                      weight_decay=weight_decay)
        # add transition
        x2PSSM = transition(x2PSSM, init_form, nb_filter, dropout_rate=dropout_rate,
                            weight_decay=weight_decay)
        x2PSSM_transpose = transition(x2PSSM_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
                                      weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x2PSSM = denseblock(x2PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    x2PSSM_transpose = denseblock(x2PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)

    x2PSSM = residual_block(x2PSSM, init_form, nb_filter, filter_size_block2, weight_decay)
    x2PSSM_transpose = residual_block(x2PSSM_transpose, init_form, nb_filter, filter_size_block2, weight_decay)

    # x2PSSM = PReLU()(x2PSSM)
    # x2PSSM_transpose = PReLU()(x2PSSM_transpose)
    x2PSSM = Activation('relu')(x2PSSM)
    x2PSSM_transpose = Activation('relu')(x2PSSM_transpose)

    # x2 = two_head_attention_fusion(x2,x2_transpose,2)

    # third input seq of 15 #
    inputPSSM3 = Input(shape=img_PSSMdim3)
    inputPSSM3_transpose = Permute((2, 1))(inputPSSM3)
    x3PSSM = Conv1D(nb_filter, filter_size_ori,
                    kernel_initializer=init_form,
                    activation='relu',
                    padding='same',
                    use_bias=False,
                    kernel_regularizer=L2(weight_decay))(inputPSSM3)
    x3PSSM_transpose = Conv1D(nb_filter, filter_size_ori,
                              kernel_initializer=init_form,
                              activation='relu',
                              padding='same',
                              use_bias=False,
                              kernel_regularizer=L2(weight_decay))(inputPSSM3_transpose)
    # x3PSSM = transformer_block(inputPSSM3, nb_filter)
    # x3PSSM_transpose = transformer_block(inputPSSM3_transpose,nb_filter)
    # x3PSSM = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x3PSSM)
    # x3PSSM_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x3PSSM_transpose)
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x3PSSM = denseblock(x3PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                            dropout_rate=dropout_rate,
                            weight_decay=weight_decay)
        x3PSSM_transpose = denseblock(x3PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate,
                                      filter_size_block3,
                                      dropout_rate=dropout_rate,
                                      weight_decay=weight_decay)
        # add transition
        x3PSSM = transition(x3PSSM, init_form, nb_filter, dropout_rate=dropout_rate,
                            weight_decay=weight_decay)
        x3PSSM_transpose = transition(x3PSSM_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
                                      weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x3PSSM = denseblock(x3PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    x3PSSM_transpose = denseblock(x3PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
    x3PSSM = residual_block(x3PSSM, init_form, nb_filter, filter_size_block3, weight_decay)
    x3PSSM_transpose = residual_block(x3PSSM_transpose, init_form, nb_filter, filter_size_block3, weight_decay)

    x3PSSM = Activation('relu')(x3PSSM)
    x3PSSM_transpose = Activation('relu')(x3PSSM_transpose)
    # x3PSSM = PReLU()(x3PSSM)
    # x3PSSM_transpose = PReLU()(x3PSSM_transpose)

    # x3 = two_head_attention_fusion(x3, x3_transpose,2)

    # x=concatenate([x1,x2,x3],axis=1)
    # 需要改进!!!

    # x = Flatten()(attention)  有问题
    # x = attention_block(x)
    xPSSM = concatenate([x1PSSM, x2PSSM, x3PSSM], axis=1)
    xPSSM_transpose = concatenate([x1PSSM_transpose, x2PSSM_transpose, x3PSSM_transpose], axis=2)
    # xPSSM = multi_head_attention_fusion(x1PSSM, x2PSSM, x3PSSM, num_heads=6, Daxis=1)
    # xPSSM_transpose = multi_head_attention_fusion(x1PSSM_transpose, x2PSSM_transpose, x3PSSM_transpose, num_heads=6, Daxis=2)
    x = concatenate([x, xPSSM], axis=-1)
    x_t = concatenate([x_transpose, xPSSM_transpose], axis=1)
    # x = two_head_attention_fusion(x,xPSSM,2,axis=-1)
    # x_t = two_head_attention_fusion(x_transpose, xPSSM_transpose, 2, axis=1)

    #
    # x = two_head_attention_fusion(x,x_transpose,4,axis=1)
    # x = concatenate([x,xPSSM],axis=-1)
    # x_t = two_head_attention_fusion(x_transpose,xPSSM_transpose,2,axis=1)
    # x_t = concatenate([x_transpose,xPSSM_transpose],axis=1)
    x_t_last = x_t.shape[-1]
    x_last_shape = x.shape[-1]
    x_t = Conv1D(filters=x_last_shape, kernel_size=1)(x_t)  # 可改进
    # x = Conv1D(filters=x_t_last,kernel_size=1)(x)
    # x = concatenate([x,x_t],axis=1)
    x = concatenate([x, x_t], axis=1)
    # x = two_head_attention_fusion(x,x_t,num_heads=6,axis=1)

    x = Flatten()(x)

    x = Dense(dense_number,
              name='Dense_1',
              activation='relu',
              kernel_initializer=init_form,
              kernel_regularizer=L2(weight_decay),
              bias_regularizer=L2(weight_decay))(x)  # unsure a probleming point

    x = Dropout(dropout_dense)(x)
    # softmax

    output1 = Dense(nb_classes, activation='softmax', kernel_initializer=init_form,
                    kernel_regularizer=L2(weight_decay),
                    bias_regularizer=L2(weight_decay))(x)

    methy_model = Model(inputs=[main_input, input2, input3, main_PSSMinput, inputPSSM2, inputPSSM3], outputs=[output1],
                        name="multi-DenseNet")

    return methy_model

def model_net(X_BLtest1,X_BLtest2,X_BLtest3,X_PSSMtest1,X_PSSMtest2,X_PSSMtest3,y_test,win1,win2,win3,sites,
              nb_epoch ,coding):

    nb_classes = 2



    ##########parameters#########



    nb_classes = 2
    img_BLdim1 = X_BLtest1.shape[1:]
    img_BLdim2 = X_BLtest2.shape[1:]
    img_BLdim3 = X_BLtest3.shape[1:]
    img_PSSMdim1 = X_PSSMtest1.shape[1:]
    img_PSSMdim2 = X_PSSMtest2.shape[1:]
    img_PSSMdim3 = X_PSSMtest3.shape[1:]

    X_BLtest1 = X_BLtest1.astype(np.float16)
    X_BLtest2 = X_BLtest2.astype(np.float16)
    X_BLtest3 = X_BLtest3.astype(np.float16)
    X_PSSMtest1 = X_PSSMtest1.astype(np.float16)
    X_PSSMtest2 = X_PSSMtest2.astype(np.float16)
    X_PSSMtest3 = X_PSSMtest3.astype(np.float16)

    y_test = y_test.astype(np.float16)
    init_form = 'RandomUniform'  # RandomUniform
    learning_rate = 0.001  # 0.001->0.01
    nb_dense_block = 1  # 1
    # nb_layers = 5
    nb_layers = 5
    nb_filter = 32
    growth_rate = 32
    # growth_rate = 24
    filter_size_block1 = 13  # 13
    filter_size_block2 = 7  # 7
    filter_size_block3 = 3  # 3
    filter_size_ori = 1
    dense_number = 32
    dropout_rate = 0.2  # 0.2
    dropout_dense = 0.3  # 0.3
    weight_decay = 0.0001  # 0.0001->0.001

    nb_batch_size = 512  # 512->256


    model = Methys(nb_classes, nb_layers, img_BLdim1, img_BLdim2, img_BLdim3, img_PSSMdim1, img_PSSMdim2, img_PSSMdim3,init_form, nb_dense_block,
                             growth_rate, filter_size_block1, filter_size_block2, filter_size_block3,
                             nb_filter, filter_size_ori,
                             dense_number, dropout_rate, dropout_dense, weight_decay)


    ###################
    # Construct model #
    ###################
    # from methods.Methy import Methys




    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    l2_lambda=0.01

    regularization_loss = 0.0
    for weight in model.trainable_weights:
        regularization_loss += tf.nn.l2_loss(weight)

    regularization_loss *= l2_lambda

    def combined_loss(y_true, y_pred, regularization_loss=regularization_loss):
        # Calculate weighted binary crossentropy loss
        wbce_loss = weighted_binary_crossentropy(y_true, y_pred)
        # Combine with regularization loss
        total_loss = wbce_loss + regularization_loss
        return total_loss

    # model compile loss!!
    model.compile(loss=weighted_binary_crossentropy,
                    optimizer=opt,
                    metrics=['accuracy'])
    # y_train_squeezed = (np.argmax(y_train,axis=1)).astype(int)
    #
    # # 计算类别权重
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_squeezed),
    #                                      y=y_train_squeezed)
    # class_weights = dict(enumerate(class_weights))



    return model
def load_blosum62_matrix(file_path):
    """从 CSV 文件加载 BLOSUM62 矩阵"""
    df = pd.read_csv(file_path, index_col=0)
    blosum62 = df.to_dict()  # 将数据框转换为字典
    amino_acids = df.columns.tolist()  # 提取氨基酸列表
    return blosum62, amino_acids

def get_blosum62_row(aa, df,valid_amino_acids):
    """获取给定氨基酸的 BLOSUM62 矩阵行向量"""
    if aa not in df.index:
        return [0] * len(df.columns)
    row = df.loc[aa]
        # 保留有效氨基酸的得分，其它设置为0
    return [row.get(other, 0) if other in valid_amino_acids else 0 for other in df.columns]

def encode_sequence(sequence, df):
    """将蛋白质序列编码为 BLOSUM62 矩阵的特征向量"""
    valid_amino_acids = set(sequence)
    encoding = []
    for aa in sequence:
        if aa =='X':
            aa = 'O'
        if aa in df.index:
            row = get_blosum62_row(aa, df,valid_amino_acids)
            encoding.append(row)
        else:
            raise ValueError(f"Unknown amino acid: {aa}")
    return np.array(encoding)

def BLOSUM_Encode(train_file,window_size):
    pos = []  # list of position with protein name
    rawseq = []
    all_label = []
    short_seqs = []
    half_len = int((window_size - 1) / 2)
    sites = 'S'

    empty_aa = '-'  # 空白氨基酸，用于填充序列不足的情况
    import csv
    file_path = 'D:\Code_Implentation\DeepMethy\\blosum.csv'
    df = pd.read_csv(file_path, index_col=0)
    with open(train_file, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
            position = int(row[1])
            sseq = row[2]
            rawseq.append(row[2])
            center = sseq[position - 1]
            if (center in sites) or (center in 'T') or (center in 'Y'):
                all_label.append(int(row[0]))
                pos.append(row[1])

                # 提取窗口内的氨基酸序列
                if position - half_len > 0:
                    start = int(position - half_len)
                    left_seq = sseq[start - 1:position - 1]
                else:
                    left_seq = sseq[0:position - 1]

                end = len(sseq)
                if position + half_len < end:
                    end = int(position + half_len)
                right_seq = sseq[position:end]

                # 处理序列不足窗口大小的情况
                if len(left_seq) < half_len:
                    nb_lack = half_len - len(left_seq)
                    left_seq = ''.join([empty_aa for count in range(nb_lack)]) + left_seq

                if len(right_seq) < half_len:
                    nb_lack = half_len - len(right_seq)
                    right_seq = right_seq + ''.join([empty_aa for count in range(nb_lack)])

                # 将窗口序列编码为BLOSUM
                shortseq = left_seq + center + right_seq
                encoded_seq = []
                encoded_seq.append(encode_sequence(shortseq, df))
            short_seqs.append(encoded_seq)


    # 转换标签为one-hot编码
    targetY = to_categorical(all_label)
    import numpy as np
    # 将编码后的序列转换为numpy数组
    Matr = np.array(short_seqs)
    Matr = K.squeeze(Matr, axis=1)
    Matr = np.array(Matr)
    return Matr,targetY

def predict_for_deepmeths(sites,coding,
                         hierarchy=None, kinase=None):
    '''

    :param train_file_name: input of your prdict file
                            it must be a .csv file and theinput format  is proteinName, postion,sites, shortseq
    :param sites: the sites predict: site = 'S','T' OR 'Y'
    :param predictFrame: 'general' or 'kinase'
    :param hierarchy: if predictFrame is kinse: you must input the hierarchy:
            group,family,subfamily,kinase to choose corresponding model
    :param kinase: kinase name
    :return:
     a file with the score
    '''

    import numpy as np
    # win1 = 51->41(xin)
    win1 = 33
    win2 = 20
    win3 = 9

    fold_path = 'D:/Code_Implentation/DeepMethy/dataset/Y_test.csv'

    X_BLtest1,y_test = BLOSUM_Encode(fold_path,win1)
    X_BLtest2,_ = BLOSUM_Encode(fold_path,win2)
    X_BLtest3,_ = BLOSUM_Encode(fold_path,win3)

    folder_path = 'D:/Code_Implentation/DeepMethy/phos/Y-test'

    X_PSSMtest1 = (process_all_pssm_files(folder_path, window_size=win1)).numpy()
    X_PSSMtest2 = (process_all_pssm_files(folder_path, window_size=win2)).numpy()
    X_PSSMtest3 = (process_all_pssm_files(folder_path, window_size=win3)).numpy()


    import numpy as np
    # from methods.model_n import model_net
    model = model_net(X_BLtest1, X_BLtest2, X_BLtest3,X_PSSMtest1,X_PSSMtest2,X_PSSMtest3, y_test,win1=41,win2=25,win3=11,sites='R', nb_epoch=0, coding=coding)

    outputfile = 'D:/Code_Implentation/DeepMethy/result/general_{:s}_Code({})'.format(site,coding)
    # model_weight = './models/model_general_{:s}_win({:d},{:d},{:d})_Code({}).weights.h5'.format(site,win1,win2,win3,coding)
    model_weight="D:/Code_Implentation/DeepMethy/models/model_Y_Code({}).weights.h5".format(coding)
    model.load_weights(model_weight)

    X_BLtest1 = X_BLtest1.astype(np.float16)
    X_BLtest2 = X_BLtest2.astype(np.float16)
    X_BLtest3 = X_BLtest3.astype(np.float16)
    X_PSSMtest1 = X_PSSMtest1.astype(np.float16)
    X_PSSMtest2 = X_PSSMtest2.astype(np.float16)
    X_PSSMtest3 = X_PSSMtest3.astype(np.float16)
    y_test = y_test.astype(np.float16)
    X_BL = concatenate([X_BLtest1,X_BLtest2,X_BLtest3],axis=1)
    X_PSSM = concatenate([X_PSSMtest1, X_PSSMtest2, X_PSSMtest3], axis=1)
    X_test = np.concatenate([X_BL,X_PSSM],axis=-1)


    predictions_R = model.predict([X_BLtest1, X_BLtest2, X_BLtest3, X_PSSMtest1, X_PSSMtest2, X_PSSMtest3])
    # results_R = np.column_stack((position1,predictions_R[:, 1]))
    results_R = predictions_R[:, 1]

    result = pd.DataFrame(predictions_R[:, 1])
    result.to_csv(outputfile + "_prediction_ arginine_methylation.txt", index=False, header=None, sep='\t',
                  quoting=csv.QUOTE_NONNUMERIC)
    pred_label=predictions_R[:,1]
    pred = pred_label
    for i in range(pred_label.shape[0]):
        if(pred_label[i]>=0.5):
            pred_label[i]=int(1)
        else:
            pred_label[i]=int(0)
    pred_label.astype(int)

    true_label=y_test[:,1]
    for i in range(true_label.shape[0]):
        if(true_label[i]>=0.5):
            true_label[i]=int(1)
        else:
            true_label[i]=int(0)
    true_label.astype(int)

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE
    # from keras.src.models import Model
    #
    # # 假设 model 是你已经训练好的 Keras 模型
    # # 获取特征嵌入层的输出，这里假设是倒数第二层
    # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=model.layers[-2].output)
    #
    # # 从数据集中获取嵌入和标签
    # embeddings = intermediate_layer_model.predict([X_BLtest1,X_BLtest2,X_BLtest3,X_PSSMtest1,X_PSSMtest2,X_PSSMtest3])
    # labels = np.argmax(y_test, axis=1)
    #
    # # 使用 t-SNE 将高维特征降到 2D 空间
    # tsne = TSNE(n_components=2, random_state=42)
    # embeddings_2d = tsne.fit_transform(embeddings)
    #
    # # 可视化
    # plt.figure(figsize=(8, 6))
    # for label in np.unique(labels):
    #     idx = labels == label
    #     plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f'Label {label}', s=5)
    # # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=5)
    #
    # plt.title('DeepMethy')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.legend()
    # plt.savefig('t-SNE DeepMethy.png')
    # plt.show()
    #
    # layer_weights = model.layers[-1].get_weights()[0]
    #
    # plt.figure(figsize=(10, 8))
    # plt.imshow(layer_weights, cmap='viridis', aspect='auto')
    # plt.colorbar()
    # plt.title('DeepMethy')
    # plt.savefig('Heatmap DeepMethy.png')
    # plt.show()
    #
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # 假设我们有测试集 X_test 和真实标签 y_test
    # # 以及模型对测试集的预测值
    # y_pred = model.predict([X_BLtest1,X_BLtest2,X_BLtest3,X_PSSMtest1,X_PSSMtest2,X_PSSMtest3])
    #
    # # 计算残差
    # residuals = y_pred - y_test
    #
    # # 绘制残差的直方图
    # plt.figure(figsize=(8, 6))
    # plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    # plt.title('DeepMethy')
    # plt.xlabel('Residual')
    # plt.ylabel('Frequency')
    # plt.savefig('Residual analysis DeepMethy.png')
    # plt.show()

    # 如果要对比消融前后的残差，假设有消融模型的预测值 y_pred_ablated
    # y_pred_ablated = ablated_model.predict(X_test)
    # residuals_ablated = y_pred_ablated - y_test
    #
    # # 绘制对比图
    # plt.figure(figsize=(8, 6))
    # plt.hist(residuals, bins=30, alpha=0.5, label='Original', color='blue', edgecolor='k')
    # plt.hist(residuals_ablated, bins=30, alpha=0.5, label='Ablated', color='red', edgecolor='k')
    # plt.title('Residuals Before and After Ablation')
    # plt.xlabel('Residual')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()

    # X_all = np.concatenate([X_BLtest1, X_BLtest2, X_BLtest3, X_PSSMtest1, X_PSSMtest2, X_PSSMtest3], axis=1)
    #

    # # 进行 PCA 降维并可视化
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_all)
    #
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=true_label, cmap='viridis', edgecolor='k', s=50)
    # plt.title("PCA Visualization")
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    # plt.colorbar(label='Classes')
    # plt.show()


    # 从代码中获取预测标签和真实标签
    # 假设 pred_label 和 true_label 已经在上述代码中计算出

    # 生成散点图


    cm = confusion_matrix(true_label, pred_label)
    # 计算准确率
    accuracy = accuracy_score(true_label, pred_label)
    print("Accuracy:", accuracy)

    # 计算精确率
    precision = precision_score(true_label, pred_label, average='weighted')
    print("Precision:", precision)

    # 计算召回率
    recall = recall_score(true_label, pred_label, average='weighted')
    print("Recall:", recall)

    # 计算F1分数
    f1 = f1_score(true_label, pred_label, average='weighted')
    print("F1 Score:", f1)


    TP=cm[0][0]
    FN=cm[0][1]
    FP=cm[1][0]
    TN=cm[1][1]



    #计算MCC
    MCC=(TP*TN-FN*FP)/math.sqrt((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))
    print("MCC score is:",MCC)



    # plt.ioff()
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    import numpy as np
    #
    results=results_R
    results=results.astype(float)
    # thresholds = np.linspace(0, 1, 10000)  # 生成 0 到 1 之间的 100 个阈值
    # fprs = []
    # tprs = []
    # precisions = []
    # recalls = []
    #
    # for thresh in thresholds:
    #     # 根据当前阈值生成二值化的预测结果
    #     y_pred = (results >= thresh).astype(int)
    #
    #     # 计算 FPR, TPR, Precision, Recall
    #     fp = np.sum((y_pred == 1) & (true_label == 0))
    #     tp = np.sum((y_pred == 1) & (true_label == 1))
    #     fn = np.sum((y_pred == 0) & (true_label == 1))
    #     tn = np.sum((y_pred == 0) & (true_label == 0))
    #
    #     fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    #     tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    #
    #     precision = precision_score(true_label, results, zero_division=0)
    #     recall = recall_score(true_label, results, zero_division=0)
    #
    #     fprs.append(fpr)
    #     tprs.append(tpr)
    #     precisions.append(precision)
    #     recalls.append(recall)
    fprs, tprs, thresholds = roc_curve(true_label, results)
    Precision, Recall, thresholds  = precision_recall_curve(true_label,results)
    #
    np.save("D:/Code_Implentation/DeepMethy/npy/Y/fpr_general_{:s}_win({:d},{:d},{:d}_Code({}).npy".format(site,win1,win2,win3,coding), fprs)
    np.save("D:/Code_Implentation/DeepMethy/npy/Y/tpr_general_{:s}_win({:d},{:d},{:d}_Code({}).npy".format(site,win1,win2,win3,coding), tprs)

    np.save("D:/Code_Implentation/DeepMethy/npy/Y/precision_general_{:s}_win({:d},{:d},{:d}_Code({}).npy".format(site,win1,win2,win3,coding), Precision)
    np.save("D:/Code_Implentation/DeepMethy/npy/Y/recall_general_{:s}_win({:d},{:d},{:d}_Code({}).npy".format(site,win1,win2,win3,coding), Recall)
    print("save .npy done")
    pr_auc = auc(Recall, Precision)
    #
    roc_auc = auc(fprs, tprs)
    print("rocauc",roc_auc)
    print("prauc",pr_auc)
    #
    # # 加载.npy文件中的数据
    fprs = np.load("D:/Code_Implentation/DeepMethy/npy/Y/fpr_general_{:s}_win({:d},{:d},{:d}_Code({}).npy".format(site,win1,win2,win3,coding))
    tprs = np.load("D:/Code_Implentation/DeepMethy/npy/Y/tpr_general_{:s}_win({:d},{:d},{:d}_Code({}).npy".format(site,win1,win2,win3,coding))
    Precision = np.load("D:/Code_Implentation/DeepMethy/npy/Y/precision_general_{:s}_win({:d},{:d},{:d}_Code({}).npy".format(site,win1,win2,win3,coding))
    Recall = np.load("D:/Code_Implentation/DeepMethy/npy/Y/recall_general_{:s}_win({:d},{:d},{:d}_Code({}).npy".format(site,win1,win2,win3,coding))

    # # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc(fprs, tprs)))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('D:/Code_Implentation/DeepMethy/dsvisualization/Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("D:/Code_Implentation/DeepMethy/visualization/ROC_win({},{},{}_Code({}).png".format(win1,win2,win3,coding))
    plt.show()

    #
    # # 绘制精确率-召回率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(Recall, Precision, color='green', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    #plt.grid(True)
    plt.savefig("D:/Code_Implentation/DeepMethy/visualization/Precision-Recall_win({},{},{}_Code({}).png".format(win1,win2,win3,coding))
    plt.show()

    #
    # # Metrics values
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
    metrics_values = [accuracy, precision, recall, f1, MCC]
    evaluate=pd.DataFrame([metrics_values],columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC'])
    evaluate.to_csv("D:/Code_Implentation/DeepMethy/dataset/evaluate_win({},{},{})_Code({}).csv".format(win1,win2,win3,coding),index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.ylim(0, 1)  # Adjust y-axis limits if needed
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("D:/Code_Implentation/DeepMethy/visualization/Evaluate_Metric_win({},{},{})_Code({}).png".format(win1,win2,win3,coding))
    plt.show()



    print("figure is Done!")

# 文件路径
if __name__ == '__main__':
    site='Y'
    code='BLOSUM_PSSM_Y'
    predict_for_deepmeths(site,coding=code)
