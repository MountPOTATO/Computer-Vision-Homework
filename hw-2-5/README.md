# Bird's Eye View Generation

## Environment Requirements

* Language：python>=3.7
* Requirements：
  * cv2 
  * numpy 



## Files for assignment submission

* the intrinstic params are in `./intrinsic/intrinsic.txt`
* the input image is `./img/input.jpg`
* images for calibration are stored in `./img/data`
* the output bev image is `./img/output_bev.jpg`



## How to Run

change the directory to  `assmt_2/hw-2-5`：

```shell
cd assmt_2/hw-2-5
```

make sure your visual environment meet the above requirements.

### Calibration

put all the images for calibration in `./input/data` in `jpg` ,then enter


```shell
python calib.py
```

to run the calibration script, the results of calibration intrinstic params (K,Distortion) will be shown and stored in `./intrinsic` as `.npy`. 



### Bird's Eye View Generation

put the input image shot by the calibrated camera in `./img` saved as `input.jpg`, **make sure you have run** `python calib.py` **before** to get the intrinsic params of your camera, then enter:

```shell
python bev.py
```

the bird's eye view image will be generated in `./img` as `output_bev.jpg`.
