# 纹理工业品缺陷检测算法部分

## 数据
本文选择的数据包含复杂纹理瓷砖和复杂纹理织物两个数据集，下面将分别介绍，如何使用作者编写的转换工具将公开数据集转换为mmdetection支持的数据集格式。

### 纹理瓷砖
1. 下载两个部分的文件，一共9.46GB，分别解压
- 第一部分 [tile_round2_train_20210204_update.zip](https://aistudio.baidu.com/aistudio/datasetdetail/70514)
- 第二部分 [tile_round2_train_20210208.zip](https://aistudio.baidu.com/aistudio/datasetdetail/70945)

2. 执行下列script，进行合并，转换为mmdetection中的BaseDetDataset格式
```bash
python tool/tile_merge_convert.py ${DATA_ROOT}/tile_round2_train_20210204_update ${DATA_ROOT}/tile_round2_train_20210208
```

### 纹理织物
[数据文件](https://aistudio.baidu.com/aistudio/datasetdetail/96643)

1. 下载5个压缩文件，一共25.6GB，分别解压在一个目录中
2. 然后执行以下bash script合并图像（Linux or WSL Terminal）
```bash
bash tools/fabric_merge.sh ${FABRIC_ROOT_DIR}
```
fabric_merge.sh会执行Python Script转换数据为COCO格式

## 算法
本文工作为选择性特征融合方法(AttentiSelective onal Feature Fusion, SFF)和联合注意力(Channel and Spatial Joint Attention Module， CSAM)，位于mmdet/models/layers，分别为feature_fusion.py和channel_spatial_joint_attention.py。

## 实验
### 纹理瓷砖
训练配置为configs/tile

### 纹理织物
训练配置为configs/fabric