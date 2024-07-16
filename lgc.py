import numpy as np
import os

def process_lgc_int_files(im_path):
    LGC_bands = 6
    LGC_lines = 2720
    LGC_columns = 3200
    if type(im_path) != list:
        im_paths = [im_path]
    else:
        im_paths = im_path
    imgs = []
    for im_path in im_paths:
        raw = np.fromfile(im_path, dtype=np.int16).reshape(LGC_bands, LGC_lines, LGC_columns)
        raw = raw[1:4, :, :].transpose(1, 2, 0)[:, :, ::-1]
        img = ((raw / raw.max(0).max(0)[np.newaxis, np.newaxis, :]) * 255).astype(np.uint8)
        imgs.append(img)
    return imgs

def process_lgc_int_files_with_threshold(im_path, threshold=16/256):
    import cv2

    LGC_bands = 6
    LGC_lines = 2720
    LGC_columns = 3200
    if type(im_path) != list:
        im_paths = [im_path]
    else:
        im_paths = im_path
    imgs = []
    for im_path in im_paths:
        raw = np.fromfile(im_path, dtype=np.int16).reshape(LGC_bands, LGC_lines, LGC_columns)
        raw = raw[1:4, :, :].transpose(1, 2, 0)[:, :, ::-1]
        img = cv2.convertScaleAbs(raw, alpha=threshold)
        imgs.append(img)
    return imgs


def process_and_save_lgc_int_files(im_path, save_path):
    LGC_bands = 6
    LGC_lines = 2720
    LGC_columns = 3200
    if type(im_path) != list:
        im_paths = [im_path]
    else:
        im_paths = im_path
    imgs = []
    for ind, im_path in enumerate(im_paths):
        raw = np.fromfile(im_path, dtype=np.int16).reshape(LGC_bands, LGC_lines, LGC_columns)
        np.save(save_path[ind], raw)
        imgs.append(raw)
    return imgs

def convert_lgc_int_to_npy(root_dir='C:\\RS datasets\\Emelyanova datasets\\LGC dataset\\data\\LGC',
                           train_base_path='.\\data\\LGC'):
    landsat_paths, modis_paths = load_paths(root_dir=root_dir)
    landsat_save_path, modis_save_path = create_save_paths(landsat_paths, modis_paths, train_base_path=train_base_path)
    process_and_save_lgc_int_files(landsat_paths, save_path=landsat_save_path);
    process_and_save_lgc_int_files(modis_paths, save_path=modis_save_path);
    return

def create_save_paths(landsat_paths, modis_paths, train_base_path='.\\data\\LGC'):
    landsat_save_path = [train_base_path + path.split('Landsat')[-1][:-3] + 'npy' for path in landsat_paths]
    modis_save_path = [train_base_path + path.split('MODIS')[-1][:-3] + 'npy' for path in modis_paths]
    [os.makedirs(path, exist_ok=True) for path in ['\\'.join(savepath.split('\\')[:-1]) for savepath in landsat_save_path]]
    return landsat_save_path, modis_save_path

def load_paths(root_dir):
    landsat_dir = os.path.join(root_dir, 'Landsat')
    modis_dir = os.path.join(root_dir, 'MODIS')
    dates = os.listdir(landsat_dir)
    landsat_data = [os.path.join(landsat_dir, date, os.listdir(os.path.join(landsat_dir, date))[0]) for date in dates]
    modis_data = [os.path.join(modis_dir, date, os.listdir(os.path.join(modis_dir, date))[0]) for date in dates]
    return landsat_data, modis_data