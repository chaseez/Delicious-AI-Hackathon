import pathlib

images_path = 'bev_classification/images'
test_path = 'bev_classification/images/test'
train_path = 'bev_classification/images/train'

if not pathlib.Path(test_path).exists():
    pathlib.Path(test_path).mkdir()

if not pathlib.Path(train_path).exists():
    pathlib.Path(train_path).mkdir()

for file in pathlib.Path(images_path).iterdir():
    if file.name == '.DS_Store' or \
        file == test_path or \
        file == train_path:
        continue

    if 'test' in file.name:
        for data in pathlib.Path(file).iterdir():
            # Only iterate over the class directories
            if 'image-datasets' == data.name or 'input' in data.name: continue

            if pathlib.Path(f'{test_path}/{data.name}').exists():
                # print(data.name)
                for f in pathlib.Path(data).iterdir():
                    pathlib.Path(f).rename(f'{test_path}/{data.name}/{f.name}')

            else:
                pathlib.Path(data).rename(f'{test_path}/{data.name}')
            # print(f'{test_path}/{data.name}')
    else:
        for data in pathlib.Path(file).iterdir():
            # Only iterate over the class directories
            if 'image-datasets' == data.name or 'input' in data.name: continue

            if pathlib.Path(f'{train_path}/{data.name}').exists():

                for f in pathlib.Path(data).iterdir():
                    pathlib.Path(f).rename(f'{train_path}/{data.name}/{f.name}')

            else:
                pathlib.Path(data).rename(f'{train_path}/{data.name}')
            # print(f'{train_path}/{data.name}')

    # Only iterate through the test and train files
    # for f in pathlib.Path(file).iterdir():
    #     print(f.name)
        
    #     else:
    #         pass