import pathlib

train_path = 'bev_classification/images/train'
val_path = 'bev_classification/images/val'

if not pathlib.Path(val_path).exists():
    pathlib.Path(val_path).mkdir()

for id in pathlib.Path(train_path).iterdir():
    if not pathlib.Path(f"{val_path}/{id.name}").exists():
        pathlib.Path(f"{val_path}/{id.name}").mkdir()

    files = [file for file in pathlib.Path(id).iterdir()]
    val_files = files[:len(files)//8]

    [pathlib.Path(f).rename(f'{val_path}/{id.name}/{f.name}') for f in val_files]
