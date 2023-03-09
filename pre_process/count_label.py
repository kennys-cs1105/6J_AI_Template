import os

if __name__ == "__main__":
    path = "./train/1368"

    # zw ld hd hs kd fb
    count = [0, 0, 0, 0, 0, 0]
    for filename in os.listdir(path):
        if not filename.endswith(".txt"):
            continue
        # print(filename)
        with open(os.path.join(path, filename), "r") as f:
            annos = f.readlines()
            for anno in annos:
                anno0 = anno.strip().split(" ")
                class_id = int(anno0[0])
                count[class_id] += 1

    print(count)
    print("Total: {}".format(sum(count)))

    # zw ld hd hs kd fb
    # 456: [5791, 189, 607, 3076, 953, 7905]
    # 912: [2581, 83, 379, 1317, 501, 3915]
    # 1368: [1750, 76, 338, 1223, 484, 3380]
    # Total: [10122, 348, 1324, 5616, 1938, 15200]

    # 456Dev: [580, 19, 62, 308, 97, 791]
    # 912Dev: [259, 9, 38, 132, 51, 392]
    # 1368Dev: [175, 8, 34, 124, 49, 340]

    # Ori: [959, 54, 239, 741, 330, 1817]
