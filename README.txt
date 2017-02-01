/train ，/val ,分别为训练集和验证集

train.txt , val.txt ，分别为训练集与验证集的标签

mtrainldb  /train文件夹内的jpg图像转换为LMDB格式的数据,其图像名与标签对应，与train.txt一致

mvalldb    /val文件夹内的jpg图像转换为LMDB格式的数据，其图像名与标签对应，与val.txt一致。