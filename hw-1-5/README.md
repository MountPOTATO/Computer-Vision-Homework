# SIFT

## Environment Requirements

* Language：python>=3.7
* Requirements：
  * cv2 
  * numpy 



## How to Run

change the directory to  `assmt_1/hw-1-5`：

```shell
cd assmt_1/hw-1-5
```

make sure your visual environment meet the above requirements, and enter：

```shell
python main.py -f input/jishi1.png -s input/jishi2.png
```

其中:

* -f : path of the first input image, there are example images in the `input` directory
* -s: path of the second input image, there are example images in the `input` directory



 you can see the following result: 

* two images with keypoints marked in the first and the second picture
* the image of the keypoints match result
* the final output image of the SIFT algorithm



the output image will be put in `result` directory



there are other example images in the `input` directory，you can type  `-f  input/xxx1.png -s input/xxx2.png`  to see other outcomes
