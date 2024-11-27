from datetime import datetime

from datasets import load_dataset
import pymysql


# Connect to the MySQL database
conn = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="kgc",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=50
)
cursor = conn.cursor()

required_rels = {
    "IsA",
    "PartOf",
    "HasA",
    "UsedFor",
    "CapableOf",
    "AtLocation",
    "Causes",
    "HasSubevent",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPrerequisite",
    "HasProperty",
    "MotivatedByGoal",
    "ObstructedBy",
    "SymbolOf",
    "DefinedAs",
    "MannerOf",
    "LocatedNear",
    "HasContext",
    "SimilarTo",
    "CausesDesire",
    "MadeOf",
    "ReceivesAction",
}

def run():
    ds = load_dataset("peandrew/conceptnet_en_simple")
    dst = ds["train"]


    # for index, data in enumerate(dst):
    for index in range(3399036, len(dst)):
        data = dst[index]
        arg1 = fetch_word(data["arg1"])
        arg2 = fetch_word(data["arg2"])
        rel = fetch_word(data["rel"])
        if rel not in required_rels:
            print("Exclude this rel", rel)
            continue

        if arg1 == arg2 or len(arg1)<=2 or len(arg2)<=2 or arg1.isdigit() or arg2.isdigit():
            print("Exclude useless row:", data)

        row, created = get_or_create(arg1, arg2, rel)
        print(index, row)


# Function to get or create a row
def get_or_create(arg1, arg2, rel):
    # Check if the row exists
    select_query = "SELECT * FROM conceptnet5 WHERE arg1 = %s AND arg2 = %s AND rel = %s"
    cursor.execute(select_query, (arg1, arg2, rel))
    row = cursor.fetchone()

    if row:
        # Row exists
        print("Row already exists:", row)
        print(datetime.now())
        return row, False
    else:
        # Insert the row if it doesn't exist
        insert_query = "INSERT INTO conceptnet5 (arg1, arg2, rel) VALUES (%s, %s, %s)"
        cursor.execute(insert_query, (arg1, arg2, rel))
        conn.commit()
        print("Row was created.")
        print(datetime.now())
        return (arg1, arg2, rel), True

def fetch_word(word):
    wl = word.split("/")
    return wl[len(wl)-1]

if __name__ == '__main__':
    run()