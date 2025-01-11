# from pretrain.pretrain import run_pretrain
from pretrain.run_preprocess import run_preprocess, run_sample
from sum.summariztion import run_summarization


def run():
    run_preprocess()
    # merge_dataset()
    # run_mini()
    # create_mini_conceptnet()
    # run_pretrain()
    # run_summarization()
    # run_sample()

if __name__ == '__main__':
    run()