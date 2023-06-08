import sys
import traceback
from queue import Queue

import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject, QDateTime
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit

from app.utils import get_file_list


# from mmyolo.utils.misc import get_file_list


class ImageReader(QObject):
    # 定义信号，用于线程向主线程发送消息
    task_log_signal = Signal(str)
    error_signal = Signal(tuple)
    task_signal = Signal(str)
    img_update_signal = Signal(QPixmap)
    result_signal = Signal(object)
    finished_signal = Signal(int)

    def __init__(self, consumer_img_path_queue: Queue, prodcutor_image_queue: Queue):
        super().__init__()
        self.img_path_queue = consumer_img_path_queue
        self.image_queue = prodcutor_image_queue

        self.is_continue = True

    def do_img_task(self) -> None:
        self.task_log_signal.emit('Starting img reading ......')

        while self.is_continue:
            if not self.img_path_queue.empty():
                img_task = self.img_path_queue.get(block=False)
                files = None
                try:
                    if isinstance(img_task, str):
                        files, _ = get_file_list(img_task)
                        self.task_log_signal.emit("img_reader check {}".format(img_task))
                    else:
                        self.task_log_signal.emit("{}: 图像路径错误".format(QDateTime.currentDateTime().toString()))

                except:
                    traceback.print_exc()
                    exctype, value = sys.exc_info()[:2]
                    self.error_signal.emit((exctype, value, traceback.format_exc()))

                for file in files:
                    self.task_log_signal.emit("image path is {}".format(file))
                    while self.image_queue.full():
                        QThread.sleep(1)

                    image = cv2.imread(file)
                    if image is None:
                        self.task_log_signal.emit("image {} 不存在".format(file))
                    else:
                        self.image_queue.put(image)

                        self.task_log_signal.emit('{} IS Loaded'.format(file))
                self.task_log_signal.emit('文件{}个加载完成'.format(len(files)))
                self.finished_signal.emit(0)
            else:
                QThread.sleep(1)

                        # # 将numpy数组转换为QImage对象
                        # height, width, channels = image.shape
                        # bytesPerLine = channels * width
                        # qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
                        # # 将QImage对象转换为QPixmap对象
                        # qpixmap = QPixmap.fromImage(qimage)
                        # self.img_update_signal.emit(qpixmap)


    @Slot()
    def quit_slot(self):
        self.task_log_signal.emit('Set Quit Flag on Image Reader')
        self.is_continue = False
