import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from PIL import Image
import cv2
from collections import Counter
import dlib

from utils.del_broken import del_broken
from utils.resize import resize_and_save
from utils.face_seg import detect_face
from utils.fairness_agr import predidct_age_gender_race
from utils.blur_box import blur_box
from utils.utils import ensure_dir

import multiprocessing
from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def all_dirs(path):
    paths = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            paths.append(os.path.join(path, dir))
    return paths


def all_files(path):
    path = os.path.abspath(path)
    img_paths = []

    for root, dirs, files in os.walk(path):
        for file in files:
            img_path = os.path.join(path, file)
            if os.path.isfile(img_path):
                img_paths.append(img_path)
    return img_paths


def main(args):
    data_dir = os.path.abspath(args.data_dir)
    resize_data_dir = '/'.join(data_dir.split('/')[:-1] + [data_dir.split('/')[-1]+'_resize'])

    if args.del_broken or args.resize>0:
        print('-'*5+'Delete broken image files'+'-'*5) if args.del_broken else None
        print('-'*5+f'Resize image files to {args.resize}'+'-'*5) if args.resize>0 else None
        for keyword_dir in tqdm(all_dirs(data_dir)):
            img_paths = all_files(keyword_dir)
            for img_path in tqdm(img_paths):
                if args.del_broken:
                    if not del_broken(img_path):
                        continue
                if args.resize > 0:
                    ensure_dir(resize_data_dir)
                    new_img_path = os.path.join(os.path.dirname(img_path).replace(data_dir, resize_data_dir),
                                                os.path.basename(img_path))
                    resize_and_save(img_path, new_img_path, new_img_size=args.resize)

    if args.face_seg:
        print('-'*5+'Face Segmentation'+'-'*5)
        # ensure model file & img path
        ensure_dir(resize_data_dir)
        ensure_dir(os.path.abspath('./face_seg'))
        dlib.DLIB_USE_CUDA = True
        print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
        print('Get Image paths.')
        img_paths = []
        for keyword_dir in tqdm(all_dirs(resize_data_dir)):
            img_paths = img_paths + all_files(keyword_dir)
        # img_paths = ['/TNSRC/crawl_unsplash_clean_resize/children/0000.jpg',
        #              '/TNSRC/crawl_unsplash_clean_resize/children/0004.jpg',
        #              '/TNSRC/crawl_unsplash_clean_resize/children/0007.jpg',
        #              '/TNSRC/crawl_unsplash_clean_resize/children/0012.jpg']
        print('Crop face(s).')
        print(len(img_paths))
        pool = multiprocessing.Pool(4)
        with tqdm(total=len(img_paths)) as pbar:
            for _ in tqdm(pool.imap_unordered(partial(detect_face, SAVE_DETECTED_AT=os.path.abspath('./face_seg')),
                                              img_paths)):
                pbar.update()
        pool.close()
        pool.join()
        # pool.map(partial(detect_face, SAVE_DETECTED_AT=os.path.abspath('./face_seg')), img_paths)
        # for img_path in tqdm(img_paths):
        #     detect_face(img_path, os.path.abspath('./face_seg'))
    if args.fairness_agr:
        print('-'*5+'Predict age/gender/race of faces (Fairness)'+'-'*5)
        predidct_age_gender_race("AGR.csv", os.path.abspath('./face_seg/faces'))
    if args.face_blur:
        print('-'*5+'Blur Faces'+'-'*5)
        bboxs = {}
        rec_path = os.path.abspath('./face_seg/rec_path/')
        print('Load face bbox.')
        for root, dirs, files in tqdm(os.walk(rec_path)):
            for file in files:
                img_path = file
                img_file_name = img_path.split('@')[-1].split('.')[0].split('_')[0]
                img_tag = img_path.split('@')[-2]
                if img_tag in file and img_file_name in file:
                    key = img_tag+'/'+img_file_name
                    if bboxs.get(key): #, default=None
                        bboxs[key].append(np.load(os.path.join(rec_path, file), allow_pickle=True).tolist())
                    else:
                        bboxs[key] = [np.load(os.path.join(rec_path, file), allow_pickle=True).tolist()]
        # print(bboxs)

        print('Blur face(s).')
        for key in tqdm(bboxs.keys()):
            tag_path = os.path.join(resize_data_dir, key.split('/')[0])
            for root, dirs, files in os.walk(tag_path):
                for file in files:
                    if key.split('/')[1] in file:
                        img_path = os.path.abspath(os.path.join(tag_path, file))
            blur_box(img_path, bboxs[key])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computer Vision Image Pre-processing.')
    parser.add_argument('-d', '--data_dir', default='/TNSRC/crawl_unsplash_clean/',
                        type=str, help='path to the image data.')
    # parser.add_argument('-n', '--new_data_dir', default='/TNSRC/crawl_unsplash_clean_resize/',
    #                     type=str, help='new path to the image data.')
    parser.add_argument('-b', '--del_broken', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='delete broken (fault) image files.')
    parser.add_argument('-r', '--resize', default=-1, type=int,
                        help='resize images with given size (max(height, width) to given size).')
    parser.add_argument('-s', '--face_seg', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='segment face(s) in images.')
    parser.add_argument('-f', '--fairness_agr', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='get age/gender/race statistics of images for checking fairness issue.')
    parser.add_argument('-l', '--face_blur', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='blur face(s) in images.')
    main(parser.parse_args())
