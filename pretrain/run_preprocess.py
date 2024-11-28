import json
from datasets import load_dataset
from sitter.kast2core import KASTParse


def run_preprocess():
    dataset = load_dataset('code-search-net/code_search_net', 'java', split='train')
    ast = KASTParse("", "java")
    ast.setup()
    result = []
    count = 0
    print("Size:", len(dataset))
    # for i, data in enumerate(dataset):
    for index in range(100, len(dataset)):
        data = dataset[index]
        print(index, data)
        code_content = "public class Test {\n"
        code_content += data['func_code_string']
        code_content += "}"
        sr_project = ast.do_parse_content(code_content)
        sr_method = None

        for program in sr_project.program_list:
            for cls in program.class_list:
                sr_method=cls.method_list[0]
                sr_method.mkg.parse_method_name(sr_method.method_name)
                sr_method.mkg.parse_concept()
                sr_method.rebuild_mkg()
                sr_method.mkg.expand_concept_edge()
                sr_method.mkg.expand_concept_node(sr_method.method_name)
                kg = sr_method.mkg.to_dict()
        new_data = {
            "id": index,
            "code": data['func_code_string'],
            "doc": data['func_documentation_string'],
            "kg":kg
        }
        result.append(new_data)
        count += 1

        if count >=100:
            # Write the list of dictionaries to a JSON file
            with open("data.json", "w") as json_file:
                for js in result:
                    json_file.write(json.dumps(js))
                    json_file.write("\n")
                json_file.close()
                count = 0


def run_sample():
    sample_path = r"sample.java"
    with open(sample_path, 'r', encoding="utf8") as f:
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
                sr_method.mkg.expand_concept_edge()
                sr_method.mkg.expand_concept_node(sr_method.method_name)
                print(sr_method)

# if __name__ == '__main__':
#     run()
    # run_sample()