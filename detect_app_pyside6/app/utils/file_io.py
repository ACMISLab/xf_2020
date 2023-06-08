# coding:utf-8
import os

from PySide6.QtGui import QImage, QPixmap
from mmengine import scandir

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def get_file_list(source_root: str) -> [list, dict]:

    is_dir = os.path.isdir(source_root)
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(source_root, IMG_EXTENSIONS, recursive=True):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_file=is_file)

    return source_file_path_list, source_type

def ndarray_to_qpixmap(img_ndarr):
    height, width, channels = img_ndarr.shape

    bytesPerLine = channels * width
    qimage = QImage(img_ndarr.data, width, height, bytesPerLine, QImage.Format_RGB888)
    # 将QImage对象转换为QPixmap对象
    qpixmap = QPixmap.fromImage(qimage)

    return qpixmap