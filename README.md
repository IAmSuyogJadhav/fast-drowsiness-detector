# Fast Drowsiness Detector
Makes use of the [Lightning Fast Object Detector](https://github.com/IAmSuyogJadhav/Lightning-Fast-Object-Detector) to accelerate the usual drowsiness detection systems.

# Pre-Requisites

## Files

Download the following files and put them in the same directory as this repository for it to work properly.

- [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/thegopieffect/computer_vision/raw/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel)
- [deploy.prototxt.txt](https://tlgur.com/d/GZ19vDN8)
- [shape_predictor_68_face_landmarks.dat](https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)

## Packages

You will need **Python 3** with following packages installed.

- `opencv-python`
- `opencv-contrib-python`

Install them using:

```bash
pip3 install opencv-python opencv-contrib-python
```

# Run

Navigate to the directory where you cloned or [downloaded](https://github.com/IAmSuyogJadhav/fast-drowsiness-detector/archive/master.zip) and extracted this repository and run:

```bash
python3 try.py
```