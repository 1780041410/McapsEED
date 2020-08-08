import tensorflow as tf
import numpy as np
import math
from capsuleLayer import *
class MCapsEED(object):
    def __init__(self,sequence_length, embedding_size, num_filters, vocab_size, iter_routing, batch_size=128,
                 num_outputs_secondCaps=1, vec_len_secondCaps=10, pre_trained=[], filter_size=1, useConstantInit=False,
                 entity_index_word_metrix=[],embedding_matrix=[],config=None):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.embeddedPosition = tf.placeholder(tf.float64, [None, config.sequenceLength, config.sequenceLength],
                                               name="embeddedPosition")
        l2_loss = tf.constant(0.0)
        self.filter_size = filter_size  # 1
        self.num_filters = num_filters  # 50
        self.sequence_length = sequence_length  # 3
        self.embedding_size = embedding_size  # 100
        self.iter_routing = iter_routing  # 1
        self.num_outputs_secondCaps = num_outputs_secondCaps  # 1
        self.vec_len_secondCaps = vec_len_secondCaps  # 10
        self.batch_size = batch_size
        self.useConstantInit = useConstantInit  # Tru
        self.input_x_head= tf.reshape(self.input_x[:, 0], [-1, 1])
        self.input_x_tail= tf.reshape(self.input_x[:, -1], [-1, 1])

        self.config = config
        with tf.name_scope('embedding'):
            self.W_1=tf.get_variable(initializer=pre_trained,name='head_rel_tail')
            self.W_2=tf.get_variable(initializer=entity_index_word_metrix,name='find_head_tail_word')
            self.W_3=tf.get_variable(initializer=embedding_matrix,name='word_embedding')

            self.embedd_chars=tf.nn.embedding_lookup(self.W_1,self.input_x)
            self.head_word=tf.nn.embedding_lookup(self.W_2,self.input_x_head)
            self.tail_word=tf.nn.embedding_lookup(self.W_2,self.input_x_tail)
            self.head_word=tf.reshape(self.head_word,[-1,config.sequenceLength])
            self.tail_word=tf.reshape(self.tail_word,[-1,config.sequenceLength])
            self.embedd_head_word=tf.cast(tf.nn.embedding_lookup(self.W_3,self.head_word),dtype=tf.float32)
            self.embedd_tail_word=tf.cast(tf.nn.embedding_lookup(self.W_3,self.tail_word),dtype=tf.float32)
        #添加位置信息
        # with tf.name_scope('position_embeddidng'):
        #     self.embeddedPosition=self._positionEmbedding()
        # self.embedd_head_word=tf.concat([self.embedd_head_word,self.embeddedPosition],-1)
        # self.embedd_tail_word=tf.concat([self.embedd_tail_word,self.embeddedPosition],-1)
        with tf.name_scope('Transformer_cnn_attention_description'):
            for i in range(config.model.numBlocks):
                with tf.name_scope("transformer-{}".format(i + 1)):
                    #维度[batch_size, sequence_length, embedding_size]
                    multiHeadAtt_head_desp = self._multiheadAttention(rawKeys= self.head_word,
                                                                      queries=self.embedd_head_word,
                                                                      keys=self.embedd_head_word)
                    #维度[batch_size, sequence_length, embedding_size]
                    self.embededWords_head_desp=self._feedForward(multiHeadAtt_head_desp,
                                                           [config.model.filters, config.model.embeddingSize])


                    rel=tf.reshape(self.embedd_chars[:,1,:],[-1,config.model.embeddingSize,1])
                    theates_head_desp = tf.nn.softmax(tf.matmul(self.embededWords_head_desp, rel), dim=1)
                    self.embededWords_head_desp = self.embededWords_head_desp * theates_head_desp

                    multiHeadAtt_tail_desp = self._multiheadAttention(rawKeys=self.tail_word,
                                                                      queries=self.embedd_tail_word,
                                                                      keys=self.embedd_tail_word)
                    self.embededWords_tail_desp = self._feedForward(multiHeadAtt_tail_desp,
                                                                    [config.model.filters,
                                                                     config.model.embeddingSize])

                    theates_tail_desp = tf.nn.softmax(tf.matmul(self.embededWords_tail_desp, rel), dim=1)
                    self.embededWords_tail_desp = self.embededWords_tail_desp * theates_tail_desp

                    #门机制结合实体与实体描述
                    self.gate_embedding_ori = tf.get_variable(name='gate_ori',
                                                              shape=[len(pre_trained), 1],
                                                              initializer=tf.random_uniform_initializer(minval=0,
                                                                                                        maxval=1))
                    self.gate_embedding = tf.nn.sigmoid(self.gate_embedding_ori, name='gate_sigmoid')

                    head_gate = tf.nn.embedding_lookup(self.gate_embedding,self.input_x_head)
                    tail_gate = tf.nn.embedding_lookup(self.gate_embedding,self.input_x_tail)
                    head_gate = tf.reshape(head_gate, [-1, 1])
                    tail_gate=  tf.reshape(tail_gate,[-1,1])
                    head_embeded=  self.embedd_chars[:,0,:]
                    tail_embeded=self.embedd_chars[:,-1,:]
                    rel_embeded=self.embedd_chars[:,1,:]
                    head_combine_headdesc=head_gate*(tf.reduce_sum(self.embededWords_head_desp,axis=1))+(1-head_gate)*head_embeded
                    tail_combine_taildesc = tail_gate * (tf.reduce_sum(self.embededWords_tail_desp, axis=1)) + (
                                1 - tail_gate) * tail_embeded
                    self.new_embedd_chars=tf.concat([head_combine_headdesc,rel_embeded,tail_combine_taildesc],axis=-1)

                    self.new_embedd_chars = tf.reshape(self.new_embedd_chars, [-1,3,embedding_size])
                    self.X = tf.expand_dims(self.new_embedd_chars, -1)

                    # 胶囊网络层
                    self.build_arch()
                    self.loss()
                    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)

                    tf.logging.info('Seting up the main structure')




    def build_arch(self):
        #The first capsule layer
        with tf.variable_scope('FirstCaps_layer'):
            self.firstCaps = CapsLayer(num_outputs_secondCaps=self.num_outputs_secondCaps, vec_len_secondCaps=self.vec_len_secondCaps,
                                    with_routing=False, layer_type='CONV', embedding_size=self.embedding_size,
                                    batch_size=self.batch_size, iter_routing=self.iter_routing,
                                    useConstantInit=self.useConstantInit, filter_size=self.filter_size,
                                    num_filters=self.num_filters, sequence_length=self.sequence_length)

            self.caps1 = self.firstCaps(self.X, kernel_size=1, stride=1)
        #The second capsule layer
        with tf.variable_scope('SecondCaps_layer'):
            self.secondCaps = CapsLayer(num_outputs_secondCaps=self.num_outputs_secondCaps, vec_len_secondCaps=self.vec_len_secondCaps,
                                    with_routing=True, layer_type='FC',
                                    batch_size=self.batch_size, iter_routing=self.iter_routing,
                                    embedding_size=self.embedding_size, useConstantInit=self.useConstantInit, filter_size=self.filter_size,
                                    num_filters=self.num_filters, sequence_length=self.sequence_length)
            self.caps2 = self.secondCaps(self.caps1)

        self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

    def loss(self):
        self.scores = tf.reshape(self.v_length, [self.batch_size, 1])
        print('self.scores:',self.scores)
        print('*'*100)

        print('self.input_y',self.input_y)
        self.predictions = tf.nn.sigmoid(self.scores)
        print("Using square softplus loss")
        losses = tf.square(tf.nn.softplus(self.scores * self.input_y))
        self.total_loss = tf.reduce_mean(losses)

    def _positionEmbedding(self, scope="positionEmbedding"):
        # 生成可训练的位置向量
        batchSize = self.config.batchSize
        sequenceLen = self.config.sequenceLength
        embeddingSize = self.config.model.embeddingSize

        # 生成位置的索引，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        positionEmbedding = np.array(
            [[pos / np.power(10000, (i - i % 2) / embeddingSize) for i in range(embeddingSize)]
             for pos in range(sequenceLen)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)

        return positionEmbedded

    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False,
                            scope="multiheadAttention"):
        # rawKeys 的作用是为了计算mask时用的，因为keys是加上了position embedding的，其中不存在padding为0的值

        numHeads = self.config.model.numHeads
        keepProp = self.config.model.keepProp

        if numUnits is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            numUnits = queries.get_shape().as_list()[-1]

        # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
        # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
        # Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。虽然在queries中也存在这样的填充词，但原则上模型的结果之和输入有关，而且在self-Attention中
        # queryies = keys，因此只要一方为0，计算出的权重就为0。
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        keyMasks = tf.tile(rawKeys, [numHeads, 1])

        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

        # tf.ones_like生成元素全为1，维度和scaledSimilary相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,
                                  scaledSimilary)  # 维度[batch_size * numHeads, queries_len, key_len]

        # 在计算当前的词时，只考虑上文，不考虑下文，出现在Transformer Decoder中。在文本分类时，可以只用Transformer Encoder。
        # Decoder是生成模型，主要用在语言生成中
        if causality:
            diagVals = tf.ones_like(maskedSimilary[0, :, :])  # [queries_len, keys_len]
            tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()  # [queries_len, keys_len]
            masks = tf.tile(tf.expand_dims(tril, 0),
                            [tf.shape(maskedSimilary)[0], 1, 1])  # [batch_size * numHeads, queries_len, keys_len]

            paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(masks, 0), paddings,
                                      maskedSimilary)  # [batch_size * numHeads, queries_len, keys_len]

        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
        weights = tf.nn.softmax(maskedSimilary)

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=keepProp)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layerNormalization(outputs)
        return outputs

    def _layerNormalization(self, inputs, scope="layerNorm"):
        # LayerNorm层和BN层有所不同
        epsilon = self.config.model.epsilon

        inputsShape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]

        paramsShape = inputsShape[-1:]

        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        # mean, variance的维度都是[batch_size, sequence_len, 1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(paramsShape))

        gamma = tf.Variable(tf.ones(paramsShape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        outputs = gamma * normalized + beta

        return outputs
    def _feedForward(self, inputs, filters, scope="multiheadAttention"):
        # 在这里的前向传播采用卷积神经网络

        # 内层
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # 外层
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
        # 维度[batch_size, sequence_length, embedding_size]
        outputs = tf.layers.conv1d(**params)

        # 残差连接
        outputs += inputs

        # 归一化处理
        outputs = self._layerNormalization(outputs)

        return outputs

    def _positionEmbedding(self, scope="positionEmbedding"):
        # 生成可训练的位置向量
        batchSize = self.config.batchSize
        sequenceLen = self.config.sequenceLength
        embeddingSize = self.config.model.embeddingSize

        # 生成位置的索引，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        positionEmbedding = np.array(
            [[pos / np.power(10000, (i - i % 2) / embeddingSize) for i in range(embeddingSize)]
             for pos in range(sequenceLen)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)

        return positionEmbedded