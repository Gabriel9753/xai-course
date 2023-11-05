import os

from dotenv import load_dotenv

load_dotenv()

# DATAPATH = "D:\\Database\\animals\\original\\animals"
DATAPATH = str(os.getenv("DATAPATH"))


def create_imagelist():
    images_labels = {}
    for d in os.listdir(os.path.join(DATAPATH, "animals")):
        for f in os.listdir(os.path.join(DATAPATH, "animals", d)):
            images_labels[os.path.join(DATAPATH, "animals", d, f)] = d

    with open(os.path.join(DATAPATH, "images_labels.txt"), "w+", encoding="utf-8") as f:
        for k, i in images_labels.items():
            f.write(f"{k} {i}\n")


if __name__ == "__main__":
    create_imagelist()
