# Scale Invariant Points Detection

## 运行要求

* 编程语言：python>=3.7
* 环境要求：
  * tqdm (用于显示个别算法环节进度)
  * cv2 (cv2只用于卷积以加快速度，即调用cv2.filter2D方法，并未使用于算法核心部分)
  * numpy 
  * matplotlib



## 运行方式

切换目录到 `code/hw-1-4` 下：

```shell
cd code/hw-1-4
```

确保虚拟环境达到运行要求后，输入：

```shell
python main.py -i input/butterfly.jpeg -s 1.1
```

其中:

* -i : 指输入图片的路径，在input文件夹中有可供使用的示例图片
* -s: 图片尺度，比如放大到图片原来的1.1倍（代码采用LoG实现，运行时间较长，建议在1-1.6倍放大下运行

 运行后，可以看到类似以下运行结果：

```
start convolution...
100%|███████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 181.07it/s]
found 17069 blobs
removing unnecessary blobs,this may take a while...
100%|████████████████████████████████████████████| 40783996/40783996 [00:51<00:00, 790318.26it/s]
remain blobs:  1016
```

表示程序运行成功

输出的结果图片存放在result文件夹中，为随机命名



你可以尝试输入其他图片的路径
