from datasets import load_dataset

from pretrain.KG import MethodKG
from sitter.ast2core import ASTParse
from sitter.kast2core import KASTParse


def run():
    dataset = load_dataset('code-search-net/code_search_net', 'java', split='train')
    ast = ASTParse("", "java")
    ast.setup()

    for i, data in enumerate(dataset):
        print(i, data)
        code_content = "public class Test {\n"
        code_content += data['func_code_string']
        code_content += "}"
        sr_project = ast.do_parse_content(code_content)
        sr_method = None

        for program in sr_project.program_list:
            for cls in program.class_list:
                sr_method=cls.method_list[0]

        mkg = MethodKG(
            code=data['func_code_string'],
            language=data['language'],
            sr_method=sr_method,
        )
        mkg.parse_tokens()
        mkg.parse_concept_nodes()
        mkg.parse_control_dependence()
        print(mkg.nodes)

def run_sample():
    sample_path = r"sample.java"
    with open(sample_path, 'r') as f:
        ast = KASTParse("", "java")
        ast.setup()
        sample = f.read()
        code_content = "public class Test {\n"
        code_content += sample
        code_content += "}"
        sr_project = ast.do_parse_content(code_content)
        sr_method = None

        for program in sr_project.program_list:
            for cls in program.class_list:
                sr_method = cls.method_list[0]
                sr_method.mkg.parse_concept()
                sr_method.rebuild_mkg()
                print(sr_method)

if __name__ == '__main__':
    # run()
    run_sample()