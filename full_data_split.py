import pathlib

# Split the train and val data into 10 files for faster training and quicker model saving
train_path = 'bev_classification/images/train'
val_path = 'bev_classification/images/val'
num_chunks = 10

# Create the train split folders
folders = [folder.name for folder in pathlib.Path(train_path).iterdir()]
for i in range(num_chunks):
    if f'train_{i}' not in folders:
        pathlib.Path(f'{train_path}/train_{i}').mkdir()

# Create the val split folders
folders = [folder.name for folder in pathlib.Path(val_path).iterdir()]
for i in range(num_chunks):
    if f'val_{i}' not in folders:
        pathlib.Path(f'{val_path}/val_{i}').mkdir()

# Loop through all the classes and evenly distribute the photos into the split folders
for bev in pathlib.Path(train_path).iterdir():
    if 'train' in bev.name: continue

    photos = [photo for photo in pathlib.Path(bev).iterdir()]
    
    incrementer = len(photos) // num_chunks
    start_splice = 0
    end_splice = incrementer
    for i in range(num_chunks):
        if not pathlib.Path(f'{train_path}/train_{i}/{bev.name}').exists():
            pathlib.Path(f'{train_path}/train_{i}/{bev.name}').mkdir()

        train_files = photos[start_splice:end_splice]
        print(f'num files:{len(train_files)} start: {start_splice} end: {end_splice}')
        [pathlib.Path(f).rename(f'{train_path}/train_{i}/{bev.name}/{f.name}') for f in train_files]
        start_splice = end_splice
        end_splice += incrementer

for bev in pathlib.Path(val_path).iterdir():
    if 'val' in bev.name: continue

    photos = [photo for photo in pathlib.Path(bev).iterdir()]
    
    incrementer = len(photos) // num_chunks
    start_splice = 0
    end_splice = incrementer
    for i in range(num_chunks):
        if not pathlib.Path(f'{val_path}/val_{i}/{bev.name}').exists():
            pathlib.Path(f'{val_path}/val_{i}/{bev.name}').mkdir()

        val_files = photos[start_splice:end_splice]
        print(f'num files:{len(val_files)} start: {start_splice} end: {end_splice}')
        [pathlib.Path(f).rename(f'{val_path}/val_{i}/{bev.name}/{f.name}') for f in val_files]
        start_splice = end_splice
        end_splice += incrementer


# Clean up extra files
for bev in pathlib.Path(train_path).iterdir():
    if 'train' in bev.name: continue

    photos = [photo for photo in pathlib.Path(bev).iterdir()]

    [pathlib.Path(f).rename(f'{train_path}/train_0/{bev.name}/{f.name}') for f in photos]

for bev in pathlib.Path(val_path).iterdir():
    if 'val' in bev.name: continue

    photos = [photo for photo in pathlib.Path(bev).iterdir()]

    [pathlib.Path(f).rename(f'{val_path}/val_0/{bev.name}/{f.name}') for f in photos]