import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=19, help="训练轮数")
    parser.add_argument("--num_val", type=int, default=2, help="多少轮之后开始验证")
    parser.add_argument("--batch_size", type=int, default=50, help="批处理大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--pretrain", type=str, default="FALSE", help="是否使用训练好的模型")
    parser.add_argument("--file_path",type=str,default="F:/junior/xia/DL/exp/data",help='文件绝对路径')

    args = parser.parse_args()
    return args