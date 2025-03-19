import json


def test():
    path = "bc_data/data.json"
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            print(data)

if __name__ == '__main__':
    test()