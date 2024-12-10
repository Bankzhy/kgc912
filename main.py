# from pretrain.pretrain import run_pretrain
from pretrain.run_preprocess import merge_dataset


def run():
    # run_preprocess()
    merge_dataset()
    # run_mini()
    # create_mini_conceptnet()
    # run_pretrain()

if __name__ == '__main__':
    run()