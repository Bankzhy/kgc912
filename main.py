# from pretrain.pretrain import run_pretrain
from pretrain.run_preprocess import run_preprocess, run_sample, report_dataset, expand_triples
from sum.summariztion import run_summarization


# def run():
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="A simple script to say hello.")

# 添加命令行参数，包括参数名称
parser.add_argument("--start", help="")
parser.add_argument("--end", help="")

# 解析命令行参数
args = parser.parse_args()
# report_dataset()
# merge_dataset()
# run_mini()
# create_mini_conceptnet()
# run_pretrain()
# run_summarization()
# run_sample()
# run_sample()
# run_preprocess(args.start, args.end)
# run_preprocess(0, 2500)

expand_triples(0, 100)

# if __name__ == '__main__':
#     run()