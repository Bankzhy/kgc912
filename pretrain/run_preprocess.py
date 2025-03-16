import csv
import json
import os
import re
from pathlib import Path

import pymysql
import requests
from datasets import load_dataset
from tqdm import tqdm

from pretrain.dataset import KGCodeDataset
from sitter.kast2core import KASTParse
from datetime import datetime

def split_variable_name(name):
    if '_' in name:
        # 处理 snake_case
        return name.split('_')
    else:
        # 处理 CamelCase
        return re.sub('([a-z])([A-Z])', r'\1 \2', name).split()

def isNotSymbol(token):
    return bool(re.match(r'^[^\d\W]+$', token))

def is_only_letters(token):
    return bool(re.fullmatch(r"[a-zA-Z]+", token))


def check_ignore(token):
    ignore = [
        "return","us", "it", "add"
    ]
    for it in ignore:
        if token.startswith(it):
            return True
    return False


def run_mini():
    ignore_expand_concept_word = [
        "is", "has", "can", "should", "get", "are", "does", "set", "add", "remove", "delete", "update", "save",
        "create", "fetch", "on", "before", "after", "to", "from", "with", "by", "of", "for", "in", "equals", "compare",
        "if", "and", "start", "stop", "load", "reload", "read", "update", "check", "render", "class", "method",
        "generate"
    ]
    PREPOSITIONS = {
        "in", "on", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "out", "over", "under", "again", "near", "the", "that", "will", "this", "all","any"
    }
    dataset = load_dataset('code-search-net/code_search_net', 'java', split='train')
    ast = KASTParse("", "java")
    ast.setup()
    result = []
    count = 0
    print("Size:", len(dataset))
    token_map = {}
    # for i, data in enumerate(dataset):
    for index in range(0, len(dataset)):
        data = dataset[index]
        tl = data["func_documentation_tokens"]
        for tt in tl:
            token = tt.lower()
            if token in ignore_expand_concept_word:
                continue
            if isNotSymbol(token) is False:
                continue

            if token in PREPOSITIONS:
                continue

            if len(token) < 3:
                continue

            if is_only_letters(token) is False:
                continue

            if check_ignore(token) is True:
                continue

            if token not in token_map.keys():
                token_map[token] = 0
            else:
                token_map[token] += 1
        print(index, data)
    top_20_keys = [key for key, value in sorted(token_map.items(), key=lambda item: item[1], reverse=True)[:200000]]
    with open('top_20_keys.txt', 'w', encoding="utf8") as file:
        for key in top_20_keys:
            file.write(key + '\n')

    print(token_map)

def get_or_create(arg1, arg2, rel, cursor, conn):
    # Check if the row exists
    select_query = "SELECT * FROM conceptnet5_small WHERE arg1 = %s AND arg2 = %s AND rel = %s"
    cursor.execute(select_query, (arg1, arg2, rel))
    row = cursor.fetchone()

    if row:
        # Row exists
        print("Row already exists:", row)
        print(datetime.now())
        return row, False
    else:
        # Insert the row if it doesn't exist
        insert_query = "INSERT INTO conceptnet5_small (arg1, arg2, rel) VALUES (%s, %s, %s)"
        cursor.execute(insert_query, (arg1, arg2, rel))
        conn.commit()
        print("Row was created.")
        print(datetime.now())
        return (arg1, arg2, rel), True

def create_mini_conceptnet():
    # Connect to the MySQL database
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="Apple3328823%",
        database="kgc",
        charset="utf8mb4",  # Use utf8mb4 for full Unicode support,
        connect_timeout=50
    )
    cursor = conn.cursor()
    small_path = r"C:\Users\zhoun\conceptnet5.csv"
    token_path = r"C:\worksapce\research\kgc912\top_20_keys.txt"
    exist_tokens = []
    with open(token_path, "r", encoding="utf8") as file:
        tokens = file.read()
        exist_tokens = tokens.split("\n")
    with open(small_path, "r", encoding="utf8") as file:
        csv_reader = csv.reader(file)
        # header = next(csv_reader)  # Skip the header row
        for row in csv_reader:
            print(row)
            if row[1] in exist_tokens and row[2] in exist_tokens:
                row, created = get_or_create(row[1], row[2], row[3], cursor, conn)


def fetch_tl(split):
    dataset = []
    path = r"C:\worksapce\research\kgc912\tl"
    for file in os.listdir(path):
        spath = os.path.join(path, file)
        if spath.endswith(split+".json"):

            with open(spath, encoding='ISO-8859-1') as f:
                lines = f.readlines()
                print("loading dataset:", split)
                for line in lines:
                    data = json.loads(line.strip())
                    new_data = {
                        "func_documentation_string":data["comment"],
                        "func_code_string":data["code"],
                        "idx": data["id"]
                    }
                    dataset.append(new_data)

    return dataset


def fetch_big_clone(split):
    dataset = []
    path = r"C:\worksapce\example\kgc\downstream\clone\dataset\small10"
    for file in os.listdir(path):
        spath = os.path.join(path, file)
        if spath.endswith(".jsonl"):
            with open(spath, encoding='ISO-8859-1') as f:
                lines = f.readlines()
                print("loading dataset:", split)
                for line in lines:
                    data = json.loads(line.strip())

                    new_data = {
                        "func_documentation_string": "",
                        "func_code_string": data["func"],
                        "idx": data["idx"]
                    }
                    dataset.append(new_data)

    return dataset


def fetch_pcsd(split):
    root= r"C:\worksapce\research\kgc912\pcsd"
    declaration_path = os.path.join(root, "data_ps.declarations." + split)
    body_path = os.path.join(root, "data_ps.bodies."+split)
    doc_path = os.path.join(root, "data_ps.descriptions."+split)
    declarations = []
    bodies = []
    docs = []
    dataset = []

    with open(declaration_path, encoding='ISO-8859-1') as f:
        lines = f.readlines()
        for line in lines:
            content = line.replace("DCNL", "\n")
            content = content.replace("DCSP", " ")
            declarations.append(content)

    with open(body_path, encoding='ISO-8859-1') as f:
        lines = f.readlines()
        for line in lines:
            content = line.replace("DCNL", "\n")
            content = content.replace("DCSP", " ")
            bodies.append(content)

    with open(doc_path, encoding='ISO-8859-1') as f:
        lines = f.readlines()
        for line in lines:
            content = line.replace("DCNL", "\n")
            content = content.replace("DCSP", " ")
            docs.append(content)

    for index, code in enumerate(declarations):
        new_data = {
            "func_documentation_string": docs[index],
            "func_code_string": declarations[index]+bodies[index],
            "idx": index
        }
        dataset.append(new_data)

    return dataset

def run_preprocess(start, end):

    # load code search net
    # dataset = load_dataset('code-search-net/code_search_net', 'java', split='test', trust_remote_code=True)
    # load tl
    # dataset = fetch_tl("train")

    # load bcb
    dataset = fetch_big_clone("train")

    # load pcsd
    # dataset = fetch_pcsd("train")


    ast = KASTParse("", "java")
    ast.setup()
    result = []
    exist_id = []
    count = 0
    error = []
    print("Size:", len(dataset))

    # with open("sum/tl_data/train.json", "r", encoding="utf8") as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         data = json.loads(line.strip())
    #         id = data["id"]
    #         exist_id.append(id)

    # with open(r"C:\worksapce\research\kgc912\tl_data_train11.json", "r", encoding="utf8") as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         data = json.loads(line.strip())
    #         id = data["id"]
    #         exist_id.append(id)

    # for i, data in enumerate(dataset):
    for index in range(int(start), int(end)):
        if index in exist_id:
            print("exist:", index)
            continue

        # try:
        data = dataset[index]
        print(index, data)
        code_content = "public class Test {\n"
        code_content += data['func_code_string']
        code_content += "}"
        sr_project = ast.do_parse_content(code_content)

        # code_content = "class Test:\n   "
        # code_content += data['func_code_string']
        # sr_project = ast.do_parse_content(code_content)

        sr_method = None

        for program in sr_project.program_list:
            for cls in program.class_list:
                if len(cls.method_list) > 0:
                    sr_method=cls.method_list[0]
                elif len(cls.constructor_list) == 1:
                    sr_method=cls.constructor_list[0]
                else:
                    print("class error")
                    continue
                sr_method.mkg.parse_method_name(sr_method.method_name)
                sr_method.mkg.parse_concept()

                try:
                    sr_method.rebuild_mkg()
                except Exception as e:
                    continue
                sr_method.mkg.expand_concept_edge()
                sr_method.mkg.expand_concept_node(sr_method.method_name)
                kg = sr_method.mkg.to_dict()

        if sr_method is None:
            continue

        new_data = {
            "id": index,
            # "idx": data["idx"],
            "code": data['func_code_string'],
            "doc": data['func_documentation_string'],
            # "code_token": data["func_code_token"],
            "kg": kg
        }
        result.append(new_data)
        count += 1
        # except Exception as e:
        #     print(e)
        #     with open("error.txt", "w") as file:
        #         file.write(str(index))
        #         file.write("\n")
        #     continue

        # if count >=100:
            # Write the list of dictionaries to a JSON file
        file_name = "bigclone_data_train_"+str(start)+"_"+str(end)+".json"
        with open(file_name, "w") as json_file:
            for js in result:
                json_file.write(json.dumps(js))
                json_file.write("\n")
            json_file.close()
                # count = 0


def run_sample():
    sample_path = r"/Users/zhang/Documents/kgc912/pretrain/sample.java"
    with open(sample_path, 'r', encoding="utf8") as f:
        ast = KASTParse("", "python")
        ast.setup()
        sample = f.read()
        code_content = "class Test:\n   "
        code_content += sample
        sr_project = ast.do_parse_content(code_content)

        sr_method = None

        for program in sr_project.program_list:
            for cls in program.class_list:
                if len(cls.method_list) > 0:
                    sr_method = cls.method_list[0]
                elif len(cls.constructor_list) == 1:
                    sr_method = cls.constructor_list[0]
                else:
                    print("class error")
                sr_method.mkg.parse_method_name(sr_method.method_name)
                sr_method.mkg.parse_concept()


def merge_dataset():
    merged_data = []
    json_files = list(Path(r"C:\worksapce\research\kgc912\dataset_small").glob("*.json"))
    exist_id = []


    for file in json_files:
        print(file)
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                if data['id'] not in exist_id:
                    merged_data.append(data)
                    exist_id.append(data['id'])
                    print("add:", data['id'])

                    # if len(exist_id) >= 1000:
                    #     break
                else:
                    continue

    # Save the combined data into a single JSON file
    with open(r"data_small.json", 'w') as json_file:
        for js in merged_data:
            # json_file.write(str(js['id']))
            json_file.write(json.dumps(js))
            json_file.write("\n")

        json_file.close()
    print(len(merged_data))


def report_dataset():
    struct_type= [
        "control_dependency",
        "data_dependency",
    ]

    syntax_type = [
        'type_of',
        "has_method",
        "has_property",
        "assignment"
    ]

    basic_concept_type = [
        "related_concept"
    ]

    expand_concept_type = [

    ]

    struct_num = 0
    syntax_num = 0
    basic_concept_num = 0
    expand_concept_num = 0

    # file = r"C:\worksapce\research\kgc912\pretrain\kg_data\data.json"
    #
    # with open(file, encoding='ISO-8859-1') as f:
    #     lines = f.readlines()
    #     print("loading dataset:")
    #     for line in tqdm(lines):
    #         # print(line)
    #         data = json.loads(line.strip())
    #         code = data["code"]
    #         doc = data["doc"]
    #         kg = data["kg"]
    #
    #         for edges in kg:
    #             if edges["type"] in struct_type:
    #                 struct_num += 1
    #             elif edges["type"] in syntax_type:
    #                 syntax_num += 1
    #             elif edges["type"] in basic_concept_type:
    #                 basic_concept_num += 1
    #             else:
    #                 expand_concept_num += 1

    args = ""

    dataset = KGCodeDataset(args=args, task="pretrain", split="train")

    for rel in dataset.structures:
        st_l = rel.split(dataset.KG_SEP_TOKEN)
        for st in st_l:
            child_l = st.split(dataset.spliter)
            if len(child_l) < 3:
                continue
            if child_l[1] in struct_type:
                struct_num += 1
            elif child_l[1] in syntax_type:
                syntax_num += 1

    for rel in dataset.nls:
        nls_l = rel.split(",")
        for nls in nls_l:
            child_l = nls.split(dataset.spliter)
            if len(child_l) < 3:
                basic_concept_num += 1
            else:
                expand_concept_num += 1

    print("struct num:", str(struct_num))
    print("syntax num:", str(syntax_num))
    print("basic concept num:", str(basic_concept_num))
    print("expand concept num:", str(expand_concept_num))


def expand_triples(start, end):
    expand_result = []
    headers = {
        'x-rapidapi-key': "c231564601mshf9ebdbfb9bf8045p14bc3ajsn58b3138b1e56",
        'x-rapidapi-host': "meta-llama-3-70b1.p.rapidapi.com",
        'Content-Type': "application/json"
    }
    file = r"C:\worksapce\research\kgc912\clone\bc_data\data.json"
    with open(file, encoding='ISO-8859-1') as f:
        lines = f.readlines()
        print("loading dataset:")
        for i in range(start, end):
            line = lines[i]
            # print(line)
            try:
                data = json.loads(line.strip())
                code = data['code']

                msg = "Please summarize the structure information and syntax information and nature language information as triples from following code (at least 10 for each information).\n"
                msg += code
                msg = {
                    "role": "user",
                    "content": msg
                }

                sdata = {
                    "model": "meta-llama/Llama-3-70b-chat-hf",
                    "temperature": 0,
                    "messages": [msg]
                }
                response = requests.post("https://meta-llama-3-70b1.p.rapidapi.com/", json=sdata, headers=headers)

                # api = APIMaster.objects.get(api_name="llama3")
                # api.api_current_count += len(ques)
                # api.save()

                result = response.json()
                result = result["choices"][0]["message"]["content"]
                result = filter_response(result)

                expand_result.append(
                    {
                        "id": data["idx"],
                        # "idx": data["idx"],
                        "code": data['code'],
                        "doc": data['doc'],
                        # "code_token": data["func_code_token"],
                        "kg": data["kg"],
                        "expand_structure": result["structure"],
                        "expand_nlp": result["nlp"]
                    }
                )

                file_name = "bigclone_data_expand_" + str(start) + "_" + str(end) + ".json"
                with open(file_name, "w") as json_file:
                    for js in expand_result:
                        json_file.write(json.dumps(js))
                        json_file.write("\n")
                    json_file.close()

                print(data)
            except Exception as e:
                print(e)
                continue


def filter_triples(text):
    match = re.search(r"\((.*?)\)", text)

    if match:
        result = match.group(1)
        return result
    else:
        return ""

def filter_response(response):
    title = ""
    text = ""
    code = ""
    code_lang = ""
    current = ""
    lines = response.split("\n")
    inCode = False

    for line in lines:
        if line.startswith("**Structure Information"):
            current = "title"
        elif line.startswith("**Syntax Information"):
            current = "text"
        elif line.startswith("**Natural Language Information"):
            current = "code"
            continue
        elif line.startswith("**"):
            current = ""

        if line == "":
            continue

        if current == "title":
            title += line
            title += "\n"
        elif current == "text":
            text += line
            text += "\n"
        elif current == "code":
            if line.startswith("```"):
                ll = line.split("```")
                if len(ll) > 1:
                    if ll[1] != "":
                        code_lang = ll[1]
                if inCode:
                    current = ""
                else:
                    inCode = True
            else:
                code += line
                code += "\n"
    title_l = title.split("**")
    title = title_l[len(title_l) - 1]
    if title.endswith("\n"):
        title = title[:-1]
    text_l = text.split("**")
    text = text_l[len(text_l) - 1]
    if text.endswith("\n"):
        text = text[:-1]

    if code.endswith("\n"):
        code = code[:-1]


    title_l = title.split("\n")
    structures = []
    for tl in title_l:
        st = filter_triples(tl)
        if st != "":
            st = st.replace(",", "")
            structures.append(st)

    text_l = text.split("\n")
    syntax = []
    for t in text_l:
        stn = filter_triples(t)
        if stn != "":
            stn = stn.replace(",", "")
            syntax.append(stn)

    code_l = code.split("\n")
    nlp = []
    for c in code_l:
        n = filter_triples(c)
        if n != "":
            n = n.replace(",", "")
            nlp.append(n)

    return {
        "structure": structures,
        "syntax": syntax,
        "nlp": nlp,
        "code_lang": code_lang
    }

# if __name__ == '__main__':
#     # run()
#     run_sample()