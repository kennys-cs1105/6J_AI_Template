import os
import shutil
import random

# zw ld hd hs kd fb
# 456: [5791, 189, 607, 3076, 953, 7905]
# 912: [2581, 83, 379, 1317, 501, 3915]
# 1368: [1750, 76, 338, 1223, 484, 3380]
# Total: [10122, 348, 1324, 5616, 1938, 15200]

# Ori: [959, 54, 239, 741, 330, 1817]


def main():
    num_456 = [5791, 189, 607, 3076, 953, 7905]
    num_912 = [2581, 83, 379, 1317, 501, 3915]
    num_1368 = [1750, 76, 338, 1223, 484, 3380]
    load_path = "./train/456"
    store_path = "./dev/456"
    num = num_456

    fs = os.listdir(load_path)
    random.shuffle(fs)
    moved_num = [0, 0, 0, 0, 0, 0]
    for name in fs:
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(load_path, name)) as f:
            annos = f.readlines()
            mv = False
            mv_num = [0, 0, 0, 0, 0, 0]
            for anno in annos:
                anno0 = anno.strip().split(" ")
                class_id = int(anno0[0])
                mv_num[class_id] += 1
                if moved_num[class_id] < num[class_id] * 0.1:
                    mv = True
            if mv:
                for c in range(0, 6):
                    moved_num[c] += mv_num[c]
                shutil.move(os.path.join(load_path, name), os.path.join(store_path, name))
                img_name = name[:-4] + ".jpg"
                shutil.move(os.path.join(load_path, img_name), os.path.join(store_path, img_name))
    print(moved_num)

    return


if __name__ == "__main__":
    main()









