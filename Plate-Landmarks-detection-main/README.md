# readme
本项目基于Pytorch_Retinaface修改，完成车牌的定位及四个关键点检测（车牌的左上，右上，右下及左下角点）。项目源码参考至https://github.com/Fanghc95/Plate-Landmarks-detection  
使用数据约为1000张图片，在验证集上的iou结果为92%
# 环境
pytorch1.2
gpu  GeForce GT 710

# 项目目录
* data 
  * config.py  模型参数设置
  * data_augment.py 数据处理
  * wider_face.py 自定义Dataset
* layers
  * modules.multibox_loss.py 自定义损失函数
* models
  * net.py 自定义网络模块
  * retinaface.py 卷积神经网络
* res  保存验证集的可视化结果
* trt
* utils
  * box_utils.py 矩阵评分方法
  * IOU.py  自定义四边形iou计算方法
  * timer.py 定义运算时长统计方法
* weights  保存训练后的模型
* convert_to_onnx.py  模型部署方法
* detect.py 原作者的模型测试方法
* model_prediction.py 自定义的模型验证方法
* train.py 模型训练

# 模型使用
* 使用model_prediction.py中的Pred类，调用pred中的main方法，传入参数为图片路径
  pred = Pred()
  pred.main(path="./image")
* 验证集测试，使用model_prediction.py中的Validation_Set_IOU方法，参数为image图片路径和label标签路径
# 数据格式
* data
  * train
    * image
    * label  

* 一张图片对应一个json，label处理结果为[x0 y0 x2 y2 x0 y0 x1 y1 x2 y2 x3 y3 0]前4位为左上右下坐标，5-12表示四点坐标，13表示样本类别
# 模型训练
* 可修改data.config中的参数
* 修改train.py中train方法的dataset方法，该方法可在data.wider_face.py中自定义

# 引用
https://github.com/Fanghc95/Plate-Landmarks-detection
感谢Fanghc95


