import copy
import json
import sys
import traceback
from queue import Queue

import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject, QDateTime
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit

from app.components.network_manager import NetworkManager
from app.config.config import cfg
from app.utils import get_file_list
import requests

# from mmyolo.utils.misc import get_file_list


class RecordUploader(QObject):
    # 定义信号，用于线程向主线程发送消息
    task_log_signal = Signal(str)
    error_signal = Signal(tuple)
    task_signal = Signal(str)
    img_update_signal = Signal(QPixmap)
    result_signal = Signal(object)
    finished_signal = Signal(int)

    def __init__(self, det_record_queue: Queue):
        super().__init__()
        self.det_record_queue = det_record_queue
        self.url = cfg.record_sync_url.value
        self.is_continue = True

    def do_upload_task(self) -> None:
        self.task_log_signal.emit('Starting img reading ......')

        while self.is_continue:
            if not self.det_record_queue.empty():
                det_record = self.det_record_queue.get(block=False)
                img_data = copy.deepcopy(det_record['img'])
                del det_record['img']
                print(det_record['bbox'])
                det_record['pipeline'] = 'DemoLine'

                files = {'file': ('image.jpg', img_data, 'image/jpeg')}

                params = dict(biz='testing')

                resp = requests.post(url='http://' + f'{self.url}' + '/jeecg-boot/sys/common/upload', data=params, files=files)
                if resp.ok:
                    oss_res = json.loads(resp.text)
                    print(oss_res)
                    print(det_record)

                    record_params = dict(
                        url=oss_res['message'],
                        pipeline='Testing03',
                        izPass=det_record['izPass'],
                        defects=dict(
                            categories=det_record['categories'],
                            image_height=det_record['image_height'],
                            image_width=det_record['image_width'],
                            bbox= det_record['bbox'],
                        )
                    )
                    headers = {'Content-Type': 'application/json'}
                    resp2 = requests.post('http://' + f'{self.url}' + '/jeecg-boot/detect/record/add', data=json.dumps(record_params), headers=headers)

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
    def _onPostReplay(self, _, res):
        if res['code'] != 200:
            self.task_log_signal.emit('图像传输失败: {}'.format(res['msg']))
        else:

            self.task_log_signal.emit('图像传输成功')

    @Slot()
    def quit_slot(self):
        self.task_log_signal.emit('Set Quit Flag on Image Reader')
        self.is_continue = False
