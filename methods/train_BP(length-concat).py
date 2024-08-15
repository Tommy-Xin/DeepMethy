import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.src.layers import Dense, Activation, Flatten, Dropout, Reshape,Input,Lambda,Bidirectional,PReLU
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
from keras.src.layers import Dense, concatenate, Layer, LSTM
from keras.src.activations import softmax
from keras.src.legacy.backend import batch_dot,sqrt,cast,int_shape
from keras.src.layers import Dense,Reshape,multiply,Permute
from keras.src.legacy.backend import int_shape
from keras.src.legacy import backend as K
from keras.src.layers import MultiHeadAttention, LayerNormalization

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
# def Methys(nb_classes, nb_layers,img_BLdim1,img_BLdim2,img_BLdim3,img_PSSMdim1,img_PSSMdim2,img_PSSMdim3, init_form, nb_dense_block,
#              growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,
#              nb_filter, filter_size_ori,
#              dense_number,dropout_rate,dropout_dense,weight_decay):
#     """ Build the DenseNet model
#
#     :param nb_classes: int -- number of classes
#     :param img_dim: tuple -- (channels, rows, columns)
#     :param depth: int -- how many layers
#     :param nb_dense_block: int -- number of dense blocks to add to end
#     :param growth_rate: int -- number of filters to add
#     :param nb_filter: int -- number of filters
#     :param dropout_rate: float -- dropout rate
#     :param weight_decay: float -- weight decay
#     :param nb_layers:int --numbers of layers in a dense block
#     :param filter_size_ori: int -- filter size of first conv1d
#     :param dropout_dense: float---drop out rate of dense
#
#     :returns: keras model with nb_layers of conv_factory appended
#     :rtype: keras model
#
#     """
#     # first input of 33 seq #
#     main_input = Input(shape=img_BLdim1)
#     main_input_transpose = Permute((2,1))(main_input)
#     #model_input = Input(shape=img_dim)
#     # Initial convolution
#     x1 = Conv1D(nb_filter, filter_size_ori,
#                       kernel_initializer=init_form,
#                       activation='relu',
#                       padding='same',
#                       use_bias=False,
#                       #W_regularizer=l2(weight_decay))(main_input)
#                       kernel_regularizer=L2(weight_decay))(main_input)
#     x1_transpose = Conv1D(nb_filter, filter_size_ori,
#                       kernel_initializer=init_form,
#                       activation='relu',
#                       padding='same',
#                       use_bias=False,
#                       #W_regularizer=l2(weight_decay))(main_input)
#                       kernel_regularizer=L2(weight_decay))(main_input_transpose)
#     # x1 = transformer_block(main_input, nb_filter)
#     # x1_transpose = transformer_block(main_input_transpose,nb_filter)
#     # x1 = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x1)
#     # x1_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x1_transpose)
#     # Add dense blocks
#     for block_idx in range(nb_dense_block - 1):
#         x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
#                                   dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#         x1_transpose = denseblock(x1_transpose, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
#                                   dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#         # add transition
#         x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
#                        weight_decay=weight_decay)
#         x1_transpose = transition(x1_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
#                        weight_decay=weight_decay)
#
#     # The last denseblock does not have a transition
#     x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
#                               dropout_rate=dropout_rate,
#                               weight_decay=weight_decay)
#     x1_transpose = denseblock(x1_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
#                               dropout_rate=dropout_rate,
#                               weight_decay=weight_decay)
#
#     x1 = residual_block(x1, init_form, nb_filter, filter_size_block1, weight_decay)
#     x1_transpose = residual_block(x1_transpose, init_form, nb_filter, filter_size_block1, weight_decay)
#
#
#     # x1 = PReLU()(x1)
#     # x1_transpose = PReLU()(x1_transpose)
#     x1 = Activation('relu')(x1)
#     x1_transpose = Activation('relu')(x1_transpose)
#
#     # x1 = two_head_attention_fusion(x1,x1_transpose,2)
#
#
#     # second input of 21 seq #
#     input2 = Input(shape=img_BLdim2)
#     input2_transpose = Permute((2,1))(input2)
#     x2 = Conv1D(nb_filter, filter_size_ori,
#                 kernel_initializer=init_form,
#                 activation='relu',
#                 padding='same',
#                 use_bias=False,
#                 kernel_regularizer=L2(weight_decay))(input2)
#     x2_transpose = Conv1D(nb_filter, filter_size_ori,
#                 kernel_initializer=init_form,
#                 activation='relu',
#                 padding='same',
#                 use_bias=False,
#                 kernel_regularizer=L2(weight_decay))(input2_transpose)
#     # x2 = transformer_block(input2, nb_filter)
#     # x2_transpose = transformer_block(input2_transpose,nb_filter)
#     # x2 = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x2)
#     # x2_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x2_transpose)
#     # Add dense blocks
#     for block_idx in range(nb_dense_block - 1):
#         x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
#                         dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x2_transpose = denseblock(x2_transpose, init_form, nb_layers, nb_filter, growth_rate,filter_size_block2,
#                                   dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#         # add transition
#         x2 = transition(x2, init_form, nb_filter, dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x2_transpose = transition(x2_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
#                        weight_decay=weight_decay)
#
#     # The last denseblock does not have a transition
#     x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
#                     dropout_rate=dropout_rate,
#                     weight_decay=weight_decay)
#     x2_transpose = denseblock(x2_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
#                               dropout_rate=dropout_rate,
#                               weight_decay=weight_decay)
#
#     x2 = residual_block(x2, init_form, nb_filter, filter_size_block2, weight_decay)
#     x2_transpose = residual_block(x2_transpose, init_form, nb_filter, filter_size_block2, weight_decay)
#
#     x2 = Activation('relu')(x2)
#     x2_transpose = Activation('relu')(x2_transpose)
#     # x2 = PReLU()(x2)
#     # x2_transpose = PReLU()(x2_transpose)
#
#     # x2 = two_head_attention_fusion(x2,x2_transpose,2)
#
#     #third input seq of 15 #
#     input3 = Input(shape=img_BLdim3)
#     input3_transpose = Permute((2,1))(input3)
#     x3 = Conv1D(nb_filter, filter_size_ori,
#                 kernel_initializer=init_form,
#                 activation='relu',
#                 padding='same',
#                 use_bias=False,
#                 kernel_regularizer=L2(weight_decay))(input3)
#     x3_transpose = Conv1D(nb_filter, filter_size_ori,
#                 kernel_initializer=init_form,
#                 activation='relu',
#                 padding='same',
#                 use_bias=False,
#                 kernel_regularizer=L2(weight_decay))(input3_transpose)
#     # x3 = transformer_block(input3, nb_filter)
#     # x3_transpose = transformer_block(input3_transpose,nb_filter)
#     # x3 = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x3)
#     # x3_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x3_transpose)
#     # Add dense blocks
#     for block_idx in range(nb_dense_block - 1):
#         x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
#                         dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x3_transpose = denseblock(x3_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
#                                   dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#         # add transition
#         x3 = transition(x3, init_form, nb_filter, dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x3_transpose = transition(x3_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
#                        weight_decay=weight_decay)
#
#     # The last denseblock does not have a transition
#     x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
#                     dropout_rate=dropout_rate,
#                     weight_decay=weight_decay)
#     x3_transpose = denseblock(x3_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
#                               dropout_rate=dropout_rate,
#                               weight_decay=weight_decay)
#     x3 = residual_block(x3, init_form, nb_filter, filter_size_block3, weight_decay)
#     x3_transpose = residual_block(x3_transpose, init_form, nb_filter, filter_size_block3,weight_decay)
#
#     x3 = Activation('relu')(x3)
#     x3_transpose = Activation('relu')(x3_transpose)
#     # x3 = PReLU()(x3)
#     # x3_transpose = PReLU()(x3_transpose)
#     # x3 = two_head_attention_fusion(x3, x3_transpose,2)
#
#     # x=concatenate([x1,x2,x3],axis=1)
# #需要改进!!!
#
#     # x = Flatten()(attention)  有问题
#     # x = attention_block(x)
#     x = concatenate([x1,x2,x3],axis=1)
#     x_transpose = concatenate([x1_transpose,x2_transpose,x3_transpose],axis=2)
#     # x = multi_head_attention_fusion(x1, x2, x3, num_heads=6, Daxis=1)
#     # x_transpose = multi_head_attention_fusion(x1_transpose,x2_transpose,x3_transpose, num_heads=6, Daxis=2) #
#
#     main_PSSMinput = Input(shape=img_PSSMdim1)
#     main_PSSMinput_transpose = Permute((2, 1))(main_PSSMinput)
#     # model_input = Input(shape=img_dim)
#     # Initial convolution
#     x1PSSM = Conv1D(nb_filter, filter_size_ori,
#                 kernel_initializer=init_form,
#                 activation='relu',
#                 padding='same',
#                 use_bias=False,
#                 # W_regularizer=l2(weight_decay))(main_input)
#                 kernel_regularizer=L2(weight_decay))(main_PSSMinput)
#     x1PSSM_transpose = Conv1D(nb_filter, filter_size_ori,
#                           kernel_initializer=init_form,
#                           activation='relu',
#                           padding='same',
#                           use_bias=False,
#                           # W_regularizer=l2(weight_decay))(main_input)
#                           kernel_regularizer=L2(weight_decay))(main_PSSMinput_transpose)
#     # x1PSSM = transformer_block(main_PSSMinput, nb_filter)
#     # x1PSSM_transpose = transformer_block(main_PSSMinput_transpose,nb_filter)
#     # x1PSSM = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x1PSSM)
#     # x1PSSM_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x1PSSM_transpose)
#     # Add dense blocks
#     for block_idx in range(nb_dense_block - 1):
#         x1PSSM = denseblock(x1PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
#                         dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x1PSSM_transpose = denseblock(x1PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
#                                   dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#         # add transition
#         x1PSSM = transition(x1PSSM, init_form, nb_filter, dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x1PSSM_transpose = transition(x1PSSM_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#
#     # The last denseblock does not have a transition
#     x1PSSM = denseblock(x1PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
#                     dropout_rate=dropout_rate,
#                     weight_decay=weight_decay)
#     x1PSSM_transpose = denseblock(x1PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
#                               dropout_rate=dropout_rate,
#                               weight_decay=weight_decay)
#
#     x1PSSM = residual_block(x1PSSM, init_form, nb_filter, filter_size_block1, weight_decay)
#     x1PSSM_transpose = residual_block(x1PSSM_transpose, init_form, nb_filter, filter_size_block1, weight_decay)
#
#     # x1PSSM = PReLU()(x1PSSM)
#     # x1PSSM_transpose = PReLU()(x1PSSM_transpose)
#     x1PSSM = Activation('relu')(x1PSSM)
#     x1PSSM_transpose = Activation('relu')(x1PSSM_transpose)
#
#     # x1 = two_head_attention_fusion(x1,x1_transpose,2)
#
#     # second input of 21 seq #
#     inputPSSM2 = Input(shape=img_PSSMdim2)
#     inputPSSM2_transpose = Permute((2, 1))(inputPSSM2)
#     x2PSSM = Conv1D(nb_filter, filter_size_ori,
#                 kernel_initializer=init_form,
#                 activation='relu',
#                 padding='same',
#                 use_bias=False,
#                 kernel_regularizer=L2(weight_decay))(inputPSSM2)
#     x2PSSM_transpose = Conv1D(nb_filter, filter_size_ori,
#                           kernel_initializer=init_form,
#                           activation='relu',
#                           padding='same',
#                           use_bias=False,
#                           kernel_regularizer=L2(weight_decay))(inputPSSM2_transpose)
#     # x2PSSM = transformer_block(inputPSSM2, nb_filter)
#     # x2PSSM_transpose = transformer_block(inputPSSM2_transpose,nb_filter)
#     # x2PSSM = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x2PSSM)
#     # x2PSSM_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x2PSSM_transpose)
#     # Add dense blocks
#     for block_idx in range(nb_dense_block - 1):
#         x2PSSM = denseblock(x2PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
#                         dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x2PSSM_transpose = denseblock(x2PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
#                                   dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#         # add transition
#         x2PSSM = transition(x2PSSM, init_form, nb_filter, dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x2PSSM_transpose = transition(x2PSSM_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#
#     # The last denseblock does not have a transition
#     x2PSSM = denseblock(x2PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
#                     dropout_rate=dropout_rate,
#                     weight_decay=weight_decay)
#     x2PSSM_transpose = denseblock(x2PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
#                               dropout_rate=dropout_rate,
#                               weight_decay=weight_decay)
#
#     x2PSSM = residual_block(x2PSSM, init_form, nb_filter, filter_size_block2, weight_decay)
#     x2PSSM_transpose = residual_block(x2PSSM_transpose, init_form, nb_filter, filter_size_block2, weight_decay)
#
#     # x2PSSM = PReLU()(x2PSSM)
#     # x2PSSM_transpose = PReLU()(x2PSSM_transpose)
#     x2PSSM = Activation('relu')(x2PSSM)
#     x2PSSM_transpose = Activation('relu')(x2PSSM_transpose)
#
#     # x2 = two_head_attention_fusion(x2,x2_transpose,2)
#
#     # third input seq of 15 #
#     inputPSSM3 = Input(shape=img_PSSMdim3)
#     inputPSSM3_transpose = Permute((2, 1))(inputPSSM3)
#     x3PSSM = Conv1D(nb_filter, filter_size_ori,
#                 kernel_initializer=init_form,
#                 activation='relu',
#                 padding='same',
#                 use_bias=False,
#                 kernel_regularizer=L2(weight_decay))(inputPSSM3)
#     x3PSSM_transpose = Conv1D(nb_filter, filter_size_ori,
#                           kernel_initializer=init_form,
#                           activation='relu',
#                           padding='same',
#                           use_bias=False,
#                           kernel_regularizer=L2(weight_decay))(inputPSSM3_transpose)
#     # x3PSSM = transformer_block(inputPSSM3, nb_filter)
#     # x3PSSM_transpose = transformer_block(inputPSSM3_transpose,nb_filter)
#     # x3PSSM = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x3PSSM)
#     # x3PSSM_transpose = Bidirectional(LSTM(units=nb_filter,return_sequences=True))(x3PSSM_transpose)
#     # Add dense blocks
#     for block_idx in range(nb_dense_block - 1):
#         x3PSSM = denseblock(x3PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
#                         dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x3PSSM_transpose = denseblock(x3PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
#                                   dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#         # add transition
#         x3PSSM = transition(x3PSSM, init_form, nb_filter, dropout_rate=dropout_rate,
#                         weight_decay=weight_decay)
#         x3PSSM_transpose = transition(x3PSSM_transpose, init_form, nb_filter, dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#
#     # The last denseblock does not have a transition
#     x3PSSM = denseblock(x3PSSM, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
#                     dropout_rate=dropout_rate,
#                     weight_decay=weight_decay)
#     x3PSSM_transpose = denseblock(x3PSSM_transpose, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
#                               dropout_rate=dropout_rate,
#                               weight_decay=weight_decay)
#     x3PSSM = residual_block(x3PSSM, init_form, nb_filter, filter_size_block3, weight_decay)
#     x3PSSM_transpose = residual_block(x3PSSM_transpose, init_form, nb_filter, filter_size_block3, weight_decay)
#
#     x3PSSM = Activation('relu')(x3PSSM)
#     x3PSSM_transpose = Activation('relu')(x3PSSM_transpose)
#     # x3PSSM = PReLU()(x3PSSM)
#     # x3PSSM_transpose = PReLU()(x3PSSM_transpose)
#
#     # x3 = two_head_attention_fusion(x3, x3_transpose,2)
#
#     # x=concatenate([x1,x2,x3],axis=1)
#     # 需要改进!!!
#
#     # x = Flatten()(attention)  有问题
#     # x = attention_block(x)
#     xPSSM = concatenate([x1PSSM,x2PSSM,x3PSSM],axis=1)
#     xPSSM_transpose = concatenate([x1PSSM_transpose,x2PSSM_transpose,x3PSSM_transpose],axis=2)
#     # xPSSM = multi_head_attention_fusion(x1PSSM, x2PSSM, x3PSSM, num_heads=6, Daxis=1)
#     # xPSSM_transpose = multi_head_attention_fusion(x1PSSM_transpose, x2PSSM_transpose, x3PSSM_transpose, num_heads=6, Daxis=2)
#     x = concatenate([x,xPSSM],axis=-1)
#     x_t = concatenate([x_transpose,xPSSM_transpose],axis=1)
#     # x = two_head_attention_fusion(x,xPSSM,2,axis=-1)
#     # x_t = two_head_attention_fusion(x_transpose, xPSSM_transpose, 2, axis=1)
#
#     #
#     # x = two_head_attention_fusion(x,x_transpose,4,axis=1)
#     # x = concatenate([x,xPSSM],axis=-1)
#     # x_t = two_head_attention_fusion(x_transpose,xPSSM_transpose,2,axis=1)
#     # x_t = concatenate([x_transpose,xPSSM_transpose],axis=1)
#     x_t_last = x_t.shape[-1]
#     x_last_shape = x.shape[-1]
#     x_t = Conv1D(filters=x_last_shape,kernel_size=1)(x_t) #可改进
#     # x = Conv1D(filters=x_t_last,kernel_size=1)(x)
#     # x = concatenate([x,x_t],axis=1)
#     x = concatenate([x,x_t],axis=1)
#     # x = two_head_attention_fusion(x,x_t,num_heads=6,axis=1)
#
#
#     x = Flatten()(x)
#
#     x = Dense(dense_number,
#               name ='Dense_1',
#               activation='relu',
#               kernel_initializer = init_form,
#               kernel_regularizer=L2(weight_decay),
#               bias_regularizer=L2(weight_decay))(x)         #unsure a probleming point
#
#     x = Dropout(dropout_dense)(x)
#     #softmax
#
#
#     output1 = Dense(nb_classes, activation='softmax', kernel_initializer=init_form,
#                    kernel_regularizer=L2(weight_decay),
#                    bias_regularizer=L2(weight_decay))(x)
#
#     methy_model = Model(inputs=[main_input,input2,input3,main_PSSMinput,inputPSSM2,inputPSSM3], outputs=[output1], name="multi-DenseNet")
#
#     return methy_model


def model_net(win1,win2,win3,sites,
              nb_epoch ,coding):

    nb_classes = 2



    ##########parameters#########


    file_path = 'D:\Code_Implentation\DeepMethy\\blosum.csv'
    train_file = 'D:\Code_Implentation\\DeepMethy\\dataset\\Y_train.csv'
    X_BLtrain1, y_train = BLOSUM_Encode(train_file, win1)
    X_BLtrain2, _ = BLOSUM_Encode(train_file, win2)
    X_BLtrain3, _ = BLOSUM_Encode(train_file, win3)
    folder_path = 'D:/Code_Implentation/DeepMethy/phos/Y-train'
    X_PSSMtrain1 = (process_all_pssm_files(folder_path, window_size=win1)).numpy()
    X_PSSMtrain2 = (process_all_pssm_files(folder_path, window_size=win2)).numpy()
    X_PSSMtrain3 = (process_all_pssm_files(folder_path, window_size=win3)).numpy()
    nb_classes = 2
    img_BLdim1 = X_BLtrain1.shape[1:]
    img_BLdim2 = X_BLtrain2.shape[1:]
    img_BLdim3 = X_BLtrain3.shape[1:]
    img_PSSMdim1 = X_PSSMtrain1.shape[1:]
    img_PSSMdim2 = X_PSSMtrain2.shape[1:]
    img_PSSMdim3 = X_PSSMtrain3.shape[1:]

    X_BLtrain1 = X_BLtrain1.astype(np.float16)
    X_BLtrain2 = X_BLtrain2.astype(np.float16)
    X_BLtrain3 = X_BLtrain3.astype(np.float16)
    X_PSSMtrain1 = X_PSSMtrain1.astype(np.float16)
    X_PSSMtrain2 = X_PSSMtrain2.astype(np.float16)
    X_PSSMtrain3 = X_PSSMtrain3.astype(np.float16)

    y_train = y_train.astype(np.float16)
    init_form = 'RandomUniform'  # RandomUniform
      # 0.001
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

    nb_batch_size = 512  # 512


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
    from keras.src.losses import SparseCategoricalCrossentropy, binary_crossentropy
    # model compile loss!!
    model.compile(loss=binary_crossentropy,
                    optimizer=opt,
                    metrics=['accuracy'])
    # y_train_squeezed = (np.argmax(y_train,axis=1)).astype(int)
    #
    # # 计算类别权重
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_squeezed),
    #                                      y=y_train_squeezed)
    # class_weights = dict(enumerate(class_weights))

    if nb_epoch > 0 :

          model.fit([X_BLtrain1,X_BLtrain2,X_BLtrain3,X_PSSMtrain1,X_PSSMtrain2,X_PSSMtrain3], y_train, batch_size=nb_batch_size,
                              epochs=nb_epoch, shuffle=True, verbose=2)
          modelname="model_{:s}_Code({})".format(sites,coding)+'.weights.h5'
          model.save_weights(filepath="D:/Code_Implentation/DeepMethy/models/{}".format(modelname),overwrite=True)  #xin


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
    with open(train_file, 'r', encoding='utf-8', errors='ignore') as rf:
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
# 文件路径
if __name__ == '__main__':
    model=model_net(win1=33,win2=20,win3=9,sites='Y',nb_epoch=50,coding='BLOSUM_PSSM_Y')