# 配置参数

class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):
    embeddingSize = 100

    filters = 50  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
    numHeads =10  # Attention 的头数
    numBlocks =1  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.9  # multi head attention 中的dropout

    dropoutKeepProb = 0.5  # 全连接层的dropout
    l2RegLambda = 0.0


class Config(object):
    sequenceLength =100 # 取了所有序列长度的均值
    batchSize = 128

    numClasses =1

    training = TrainingConfig()

    model = ModelConfig()


# 实例化配置参数对象
