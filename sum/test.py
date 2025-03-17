import os

def merge_all():
    all_data = []
    dataset_dir = r"C:\worksapce\research\kgc912\sum\pcsd"
    for file in os.listdir(dataset_dir):
        if file.startswith("pcsd_data"):
            with open(os.path.join(dataset_dir,file), encoding="utf8") as f:
                lines = f.readlines()
                all_data.extend(lines)
    with open(os.path.join(dataset_dir, "train.json"), "w",encoding="utf8") as f:
        for line in all_data:
            f.write(line)
        f.close()

if __name__ == '__main__':
    merge_all()