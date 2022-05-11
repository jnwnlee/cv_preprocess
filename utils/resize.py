# from PIL import Image
import cv2
import os
from utils.utils import ensure_dir


def image_resize(image, size=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if size is None, then return the
    # original image
    if size is None:
        return image

    # check to see if the height >= width
    if h >= w:
        # calculate the ratio of the height and construct the
        # dimensions
        r = size / float(h)
        dim = (int(w * r), size)

    # otherwise, the height <= width
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = size / float(w)
        dim = (size, int(h * r))

    # resize the image
    resized_img = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized_img

def resize_and_save(img_path, new_img_path, new_img_size):
    try:
        # im = Image.open(img_path)
        im = cv2.imread(img_path)
    except Exception as e:
        print(f'Img path.- {img_path}')
        print(e)
        return False

    im = image_resize(im, size=new_img_size)
    ensure_dir(os.path.dirname(new_img_path))
    cv2.imwrite(new_img_path, im)
    return True