"""
该文件旨在配置训练过程中的各种参数
请按照自己的需求进行添加或者删除相应属性
"""


class Training_Config(object):
    def __init__(self,
                 embedding_dimension=50,
                 pos_embedding_dimenstion=3,
                 out_dimension=64,
                 vocab_size=20000,
                 training_epoch=1,
                 momentum=0.9,
                 weight_decay=1e-3,
                 num_val=1,
                 max_sentence_length=140,
                 cuda=True,
                 label_num=44,
                 learning_rate=1e-3,
                 batch_size=32,
                 dropout=0.3):
        self.embedding_dimension = embedding_dimension  # 词向量的维度
        self.pos_embedding_dimension = pos_embedding_dimenstion # 位置编码的维度
        self.out_dimension = out_dimension # 第一个隐藏层维度
        self.vocab_size = vocab_size  # 词汇表大小
        self.epoch = training_epoch  # 训练轮数
        self.momentum = momentum # 动量
        self.weight_decay = weight_decay
        self.num_val = num_val  # 经过几轮才开始验证
        self.max_sentence_length = max_sentence_length  # 句子最大长度
        self.label_num = label_num  # 分类标签个数
        self.lr = learning_rate  # 学习率
        self.batch_size = batch_size  # 批大小
        self.cuda = cuda  # 是否用CUDA
        self.dropout = dropout  # dropout概率

