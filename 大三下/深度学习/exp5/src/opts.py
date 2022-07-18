import argparse
from pickle import FALSE


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_num", type=int, default=2, help="训练轮数")
    parser.add_argument("--hidden_size", type=int, default=100, help="隐藏层维度大小")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--max_length", type=int, default=120, help="最大长度")
    parser.add_argument("--SOS_token", type=int, default=1, help="")
    parser.add_argument("--EOS_token", type=int, default=2, help="")
    parser.add_argument("--encoder_path", type=str, default="saved_models/encoder_seq2seq_3.pt", help="编码模型地址")
    parser.add_argument("--decoder_path", type=str, default="saved_models/decoder_seq2seq_3.pt", help="解码模型地址")
    parser.add_argument("--device", type=str, default='cpu', help="")
    parser.add_argument("--embed_size", type=int, default=100, help="嵌入维度")
    parser.add_argument("--pre_trained", type=bool, default=FALSE, help="是否使用预训练模型")
    args = parser.parse_args()
    return args