# SIFT

## 运行要求

* 编程语言：python>=3.7
* 环境要求：
  * cv2 
  * numpy 



## 运行方式

切换目录到 `code/hw-1-4` 下：

```shell
cd code/hw-1-5
```

确保虚拟环境达到运行要求后，输入：

```shell
python main.py -f input/jishi1.png -s input/jishi2.png
```

其中:

* -f : 指输入第一张图片的路径，在input文件夹中有可供使用的示例图片
* -s: 指输入第二张图片的路径，在input文件夹中有可供使用的示例图片

 运行后，可以看到：

* 弹出两张输入图片标记了特征点的结果
* 弹出了两张图的特征点匹配情况
* 弹出了最后SIFT的输出结果

表示程序运行成功

输出的结果图片存放在result文件夹中，为随机命名



input文件夹中还有其他的示例图片，分别按照 `-f  input/xxx1.png -s input/xxx2.png` 的形式可以尝试其他输入
