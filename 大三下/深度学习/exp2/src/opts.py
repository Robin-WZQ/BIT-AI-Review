import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_num", type=int, default=31, help="训练轮数")
    parser.add_argument("--val_num", type=int, default=5, help="多少轮之后开始验证")
    parser.add_argument("--batch_size", type=int, default=256, help="批处理大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="惩罚项")
    parser.add_argument("--momentum", type=float, default=0.9, help="动量")
    parser.add_argument("--gamma", type=float, default=0.1, help="衰减系数")
    parser.add_argument("--step_size", type=int, default=10, help="每n个epoch更新一次学习率")

    args = parser.parse_args()
    return args