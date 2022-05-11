# from PIL import Image
import cv2
import os

def del_broken(img_path):
    try:
        # im = Image.open(img_path)
        im = cv2.imread(img_path)
    except Exception as e:
        print(f'Img Dir.- {img_path}')
        print(e)
        # remove fault files
        os.remove(img_path)
        return False
    else:
        if im is None:
            print(f'Img Dir.- {img_path}')
            print('Error - Nonetype Img')
            # remove fault files
            os.remove(img_path)
            return False
        return True