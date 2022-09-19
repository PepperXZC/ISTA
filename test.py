import os

with open("file.list.txt", 'w') as f:
    for i in range(1, 31):
        f.write("/na" + str(str(i).zfill(2)) + "_cbq.nii\n")

# print('/data/xiezhuocheng/file_list.txt')