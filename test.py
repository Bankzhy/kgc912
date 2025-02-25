from datasets import load_dataset


def run():
    ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)
    print(len(ds))

if __name__ == '__main__':
    run()