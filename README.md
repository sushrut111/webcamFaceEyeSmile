# Face, eyes, smile detector

This detector uses the video stream from your webcam and detects faces, eyes and smiles on the faces. This works with the help of opencv library.

## Steps to run

1. Clone this repository
```shell
git clone https://github.com/sushrut111/webcamFaceEyeSmile.git
```
2. Make sure your have python 3 installed (should work with python2 as well)
3. Change directory to cloned repository and install dependencies.
```shell
cd webcamFaceEyeSmile
pip install -r requirements.txt
```
4. Run the script
```shell
python webcamDetectFace.py
```
Your camera should start capturing. Try to smile naturally and say cheese!

## Notes:
1. Face and eye detection is commented out currently. You can uncomment the following lines in the script to enable it:
```py
        # frontalface = models.detectAttributesInImage(gray, "frontalface")
        # for x, y, w, h in frontalface:
        #     cv2.rectangle(captured_image, (x,y), (x+w, y+h), (0,255,0), 2)

        # eyes = models.detectAttributesInImage(gray, "eyes")
        # for x, y, w, h in eyes:
        #     cv2.rectangle(captured_image, (x,y), (x+w, y+h), (0,255,255), 2)
```
2. If the script is giving false positives or false negatives, you might want to adjust `minNeighbors` parameter in following line in the script:
```py
        smile = models.detectAttributesInImage(gray, "smile", scaleFactor=2, minNeighbors=50)

```
3. Make sure there is enough light in the surroundings.

## Haar cascade
1. The script uses the classifiers created by haar cascade and are in `./haarcascades` directory.
2. These classifiers are copied from the (opencv repository)[https://github.com/opencv/opencv/tree/master/data/haarcascades]