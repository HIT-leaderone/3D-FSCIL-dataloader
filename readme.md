## 数据准备

请参考 [ShapeNet55](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) 来进行基类数据集的获取。请参考 [CO3D](https://ai.meta.com/datasets/co3d-downloads/) 来完成初赛增量数据集的准备。



## 3D 连续学习数据集生成器

本比赛赛方提供3D 连续学习数据集生成器，仓库连接为：[https://github.com/HIT-leaderone/3D-FSCIL-dataloader](https://github.com/HIT-leaderone/3D-FSCIL-dataloader)

请先再在**'./data/ShapeNet55'**下载上文提到的ShapeNet55数据集，并在**'/data/ShapeNet55'**下载上文提到的CO3D数据集以完成数据准备工作。

之后选手可以调用 session_settings.py 中的 shapenet2co3d() 函数来获得初赛连续学习数据集。

调用该函数返回的session_maker支持以下成员函数：

- get_id2name()：返回一个list，其中第i个元素是标签为i的类别名称。

- make_session(session_id)：传入当前增量阶段的id（基类为0，增量类从1开始），返回两个可以放入DataLoader中生成数据的dataset_train, dataset_test，分别表示训练数据和测试数据：

- tot_session()：返回总训练阶段数量（增量阶段的个数+1）。

具体使用样例可以参考main.py。该dataloader的更多细节信息和具体实现请参考仓库中的datasets/CILdataset.py。

