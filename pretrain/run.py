from datasets import load_dataset

from pretrain.KG import MethodKG


def run():
    dataset = load_dataset('code-search-net/code_search_net', 'java', split='train')
    for i, data in enumerate(dataset):
        print(i, data)
        mkg = MethodKG(
            code=data['func_code_string'],
            language=data['language'],
        )
        mkg.parse_tokens()
        mkg.parse_concept_nodes()
        print(mkg.nodes)
if __name__ == '__main__':
    run()