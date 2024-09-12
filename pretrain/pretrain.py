from datasets import load_dataset

from kg.KG import MethodKG


def run():
    dataset = load_dataset('code-search-net/code_search_net', 'java', split='train')
    for i, data in enumerate(dataset):
        print(i, data)
        mkg = MethodKG(
            code=data['func_code_string'],
            language=data['language'],
        )
        mkg.parse_tokens()
        print(mkg.tokens)
if __name__ == '__main__':
    run()