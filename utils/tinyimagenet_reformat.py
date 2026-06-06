import os
from shutil import move
from os.path import join
from os import rmdir

target_folder = os.path.join('..', 'data', 'tiny-imagenet-200', 'val')

val_dict = {}
with open(join(target_folder, 'val_annotations.txt'), 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

images_dir = join(target_folder, 'images')
paths = [join(images_dir, name) for name in os.listdir(images_dir)]
for path in paths:
    file = os.path.basename(path)
    folder = val_dict[file]
    if not os.path.exists(join(target_folder, str(folder))):
        os.mkdir(join(target_folder, str(folder)))

for path in paths:
    file = os.path.basename(path)
    folder = val_dict[file]
    dest = join(target_folder, str(folder), str(file))
    move(path, dest)

os.remove(join(target_folder, 'val_annotations.txt'))
rmdir(images_dir)
print('done reformat the validation images')
