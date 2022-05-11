import dlib
import os
import numpy as np

from utils.utils import ensure_dir

def detect_face(image_path, SAVE_DETECTED_AT, default_max_size=640, size=224, padding=0.25):
    # default_max_size = 800

    if not os.path.exists('dlib_models/mmod_human_face_detector.dat'):
        raise FileExistsError('Missing file: dlib_models/mmod_human_face_detector.dat')
    if not os.path.exists('dlib_models/shape_predictor_5_face_landmarks.dat'):
        raise FileExistsError('Missing file: dlib_models/shape_predictor_5_face_landmarks.dat')

    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    # base = 2000  # largest width and height
    img = dlib.load_rgb_image(image_path)

    # old_height, old_width, _ = img.shape
    # if not max(old_height, old_width) == default_max_size:
    #     if old_width > old_height:
    #         new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    #     else:
    #         new_width, new_height = int(default_max_size * old_width / old_height), default_max_size
    #     img = dlib.resize_image(img, rows=new_height, cols=new_width)

    dets = cnn_face_detector(img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(image_path))
        no_face_path = os.path.join(SAVE_DETECTED_AT, "no_face/")
        ensure_dir(no_face_path)
        np.save(os.path.join(no_face_path, image_path.replace("/","@")[:-4]+".npy"), image_path)
    else:
        faces = dlib.full_object_detections()
        for idx, detection in enumerate(dets):
            rect = detection.rect
            rect_pts = [rect.left(), rect.top(), rect.right(), rect.bottom()]
            rec_path = os.path.join(SAVE_DETECTED_AT, "rec_path/")
            ensure_dir(rec_path)
            np.save(os.path.join(rec_path, '.'.join(image_path.replace("/","@").split('.')[:-1])
                                 + "_" + "face" + str(idx) + ".npy"), rect_pts)
            faces.append(sp(img, rect))
        images = dlib.get_face_chips(img, faces, size=size, padding = padding)
        for idx, image in enumerate(images):
            img_name = os.path.basename(image_path) # image_path.split("/")[-1]
            path_sp = img_name.split(".")
            faces_path = os.path.join(SAVE_DETECTED_AT, "faces/")
            ensure_dir(faces_path)
            face_name = os.path.join(faces_path,  image_path.split("/")[-2] + '/'
                                     + path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
            ensure_dir(os.path.dirname(face_name))
            dlib.save_image(image, face_name)
