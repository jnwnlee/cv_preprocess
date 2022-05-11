# Computer Vision Image Processing

### Tasks
- delete broken (fault) image files
- resize images (max(height, width) -> new size)
- face segmentation (dlib)
- predict age, gender, race of face(s)
- blur face(s)

### References
- FairFace
https://github.com/dchen236/FairFace
- Imagenet-Face-Obfuscation
https://github.com/princetonvisualai/imagenet-face-obfuscation/blob/main/experiments/blurring.py
  

### Usage
```python main.py -b True -r 640 -s True -f True -l True```

```
usage: main.py [-h] [-d DATA_DIR] [-b DEL_BROKEN] [-r RESIZE]               │
               [-s FACE_SEG] [-f FAIRNESS_AGR] [-l FACE_BLUR]               │
                                                                            │
Computer Vision Image Pre-processing. 

optional arguments:                                                         │
  -h, --help            show this help message and exit                     │
  -d DATA_DIR, --data_dir DATA_DIR                                          │
                        path to the image data.                             │
  -b DEL_BROKEN, --del_broken DEL_BROKEN                                    │
                        delete broken (fault) image files.                  │
  -r RESIZE, --resize RESIZE                                                │
                        resize images with given size (max(height, width)   │
                        to given size).                                     │
  -s FACE_SEG, --face_seg FACE_SEG                                          │
                        segment face(s) in images.                          │
  -f FAIRNESS_AGR, --fairness_agr FAIRNESS_AGR                              │
                        get age/gender/race statistics of images for        │
                        checking fairness issue.                            │
  -l FACE_BLUR, --face_blur FACE_BLUR                                       │
                        blur face(s) in images. 
```