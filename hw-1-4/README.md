# Scale Invariant Points Detection

## Environment Requirements

* Language：python>=3.7
* Requirements：
  * tqdm
  * cv2 (**cv2 is only applied in convolution process (cv2.filter2D)，not involved in other alogrithm's core functions**)
  * numpy 
  * matplotlib



## How to Run

change the directory to  `assmt_1/hw-1-4` :

```shell
cd assmt_1/hw-1-4
```

make sure your visual environment meet the above requirements, and enter：

```shell
python main.py -i input/butterfly.jpeg -s 1.1
```

* -i : path of the input image，there are example images in the `input` directory
* -s: scale of the output image（the algorithm is implemented using LoG, which takes time. For a short runtime, it's recommended that the scale value between 1 and 1.6

 you can see the following result: 

```
start convolution...
100%|███████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 181.07it/s]
found 17069 blobs
removing unnecessary blobs,this may take a while...
100%|████████████████████████████████████████████| 40783996/40783996 [00:51<00:00, 790318.26it/s]
remain blobs:  1016
```

the output image will be put in `result` directory



you can try other image path as an alternative input
