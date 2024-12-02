import csv
import json
import re

import pymysql
from datasets import load_dataset
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

def run_preprocess():
    dataset = load_dataset('code-search-net/code_search_net', 'java', split='train')
    ast = KASTParse("", "java")
    ast.setup()
    result = []
    count = 0
    print("Size:", len(dataset))
    # for i, data in enumerate(dataset):
    for index in range(117500, len(dataset)):
        try:
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
        except Exception as e:
            print(e)
            with open("error.txt", "w") as file:
                file.write(str(index))
                file.write("\n")
            continue

        if count >=100:
            # Write the list of dictionaries to a JSON file
            with open("datat_small_100000.json", "w") as json_file:
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
#     run_sample()