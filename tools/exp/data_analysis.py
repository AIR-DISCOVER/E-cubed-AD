import  pickle
from pathlib import Path

data_root = Path(__file__).resolve().parents[2] / 'data' / 'nuscenes'
train = (data_root / 'vad_nuscenes_infos_temporal_train.pkl').open('rb')
val = (data_root / 'vad_nuscenes_infos_temporal_val.pkl').open('rb')

content_train=pickle.load(train)
content_val=pickle.load(val)

train_len = len(content_train['infos'])
val_len = len(content_val['infos'])


for i in range(val_len):
    val_id = content_val['infos'][i]['lidar_path']
    for j in range(train_len):
        train_id = content_train['infos'][j]['lidar_path']

        if val_id == train_id:
            print("*************** there is val sample in training set ****************: ", j)


print(1)
