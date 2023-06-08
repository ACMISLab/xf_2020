# coding:utf-8
import os.path
from queue import Queue

import mmengine
import onnxruntime
from PySide6.QtCore import QThread, qDebug
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QGroupBox, QVBoxLayout, QSizePolicy, \
    QFileDialog
from qfluentwidgets import PushButton, ComboBox, PrimaryPushButton, PushSettingCard, SearchLineEdit, TextEdit
from qfluentwidgets import FluentIcon as FIF

from app.components import ImageReader, InferWorker
from app.components.record_uploader import RecordUploader
from app.config.config import cfg


class DetectWidget(QWidget):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))

        raw_img_placeholder = QPixmap('resource/undraw_data_processing_yrrv.svg')
        detect_img_placeholder = QPixmap('resource/undraw_file_searching_re_3evy.svg')

        self.imgGroup = QGroupBox('检测', self)
        self.imgGroup.setMaximumWidth(self.parent().width() * 2)
        self.imgGroup.setMaximumHeight(self.parent().width())

        # 加载第一张图片
        self.label_img = QLabel(self)
        self.label_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_img.setScaledContents(True)
        self.label_img.setPixmap(raw_img_placeholder)

        # 加载第二张图片
        self.label_img_det = QLabel(self)
        self.label_img_det.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_img_det.setScaledContents(True)
        self.label_img_det.setPixmap(detect_img_placeholder)

        self.configGroup = QGroupBox('配置', self)
        self.finishButton = PushButton('停止', self)
        self.infer_combobox = ComboBox(self)
        self.combobox2 = ComboBox(self)

        self.source_group = QGroupBox("图像采集", self)
        # self.combobox3 = ComboBox(self)
        # self.combobox4 = ComboBox(self)
        self.testButton = PushButton('测试一下', self)
        # self.testButton.setEnabled(False)

        # music folders
        # self.imageInThisPCGroup = SettingCardGroup(
        #     'Source', self)
        self.imageFolderCard = PushSettingCard(
            '选择图像',
            FIF.PHOTO,
            "检测文件",
            cfg.get(cfg.det_image_or_dir),
            self
        )
        # self.imageInThisPCGroup.addSettingCard(self.imageFolderCard)

        self.log_group = QGroupBox("日志", self)
        self.log_editor = TextEdit(self)
        # self.log_editor.setDisabled(True)

        self.__init_layout()

        self.__init_env()

        self.__init_detector()

        self.__connect_signals_slots()

    def __init_layout(self):
        layout = QHBoxLayout(self)  # 创建一个水平布局
        layout.addWidget(self.label_img)  # 将第一个标签添加到布局中
        layout.addWidget(self.label_img_det)  # 将第二个标签添加到布局中
        self.imgGroup.setLayout(layout)

        panel_layout = QHBoxLayout()

        label_hbox_layout = QVBoxLayout()
        label_hbox_layout.addWidget(self.infer_combobox)
        label_hbox_layout.addWidget(self.combobox2)
        label_hbox_layout.addWidget(self.finishButton)
        self.configGroup.setLayout(label_hbox_layout)

        panel_layout.addWidget(self.configGroup, 1)

        source_group_layout = QVBoxLayout()
        source_group_layout.addWidget(self.imageFolderCard)
        source_group_layout.addWidget(self.testButton)
        self.source_group.setLayout(source_group_layout)

        panel_layout.addWidget(self.source_group, 1)

        log_layout = QHBoxLayout()
        log_layout.addWidget(self.log_editor)
        self.log_group.setLayout(log_layout)
        panel_layout.addWidget(self.log_group, 3)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 32, 0, 0)

        main_layout.addWidget(self.imgGroup, 2)
        main_layout.addLayout(panel_layout, 1)

        self.setLayout(main_layout)

    def __init_env(self):
        infer_devices = onnxruntime.get_available_providers()
        self.infer_combobox.addItems(infer_devices)

    def __init_detector(self):

        self.image_path_queue = Queue(maxsize=4)
        self.image_queue = Queue(maxsize=8)
        self.det_record_queue = Queue()

        self.img_reader = ImageReader(consumer_img_path_queue=self.image_path_queue, prodcutor_image_queue=self.image_queue)
        self.io_thread = QThread()

        self.img_reader.moveToThread(self.io_thread)
        self.io_thread.started.connect(self.img_reader.do_img_task)
        self.io_thread.start()

        self.infer_worker = InferWorker(consumer_image_queue=self.image_queue, productor_det_record_queue=self.det_record_queue)
        self.infer_thread = QThread()

        self.infer_worker.moveToThread(self.infer_thread)
        self.infer_thread.started.connect(self.infer_worker.do_inference)
        self.infer_thread.start()

        self.record_uploader = RecordUploader(self.det_record_queue)
        self.upload_thread = QThread()

        self.record_uploader.moveToThread(self.upload_thread)
        self.upload_thread.started.connect(self.record_uploader.do_upload_task)
        self.upload_thread.start()


    def __connect_signals_slots(self):
        # self.startButton.pressed.connect(lambda : self.infer_thread.start())

        self.finishButton.pressed.connect(self.img_reader.quit_slot)
        self.finishButton.clicked.connect(self.infer_worker.quit_slot)

        self.testButton.pressed.connect(self.commit_img_task)

        self.img_reader.task_log_signal.connect(self.handle_msg_signal)
        self.img_reader.img_update_signal.connect(lambda pixmap: self.label_img.setPixmap(pixmap))
        self.img_reader.finished_signal.connect(lambda :self.testButton.setDisabled(False))

        self.infer_worker.task_complete_signal.connect(self.handle_task_complete)
        self.infer_worker.task_log_signal.connect(self.handle_msg_signal)
        self.infer_worker.task_under_dealing_signal.connect(lambda pixmap: self.label_img.setPixmap(pixmap))


        self.imageFolderCard.clicked.connect(self.__onImageFolderCardClicked)


    def commit_img_task(self):
        qDebug('Commit')
        self.testButton.setDisabled(True)
        self.image_path_queue.put(cfg.get(cfg.det_image_or_dir))



    # def update_det_marked_img(self,  marked_det_img: QPixmap):
    #
    def when_img_task_finished(self):
        self.testButton.setDisabled(False)

    def handle_msg_signal(self, msg):
        self.log_editor.append(msg)

    def handle_task_complete(self, pixmap: QPixmap):
        self.label_img_det.setPixmap(pixmap)


    def closeEvent(self, event):

        self.img_reader.quit()
        self.infer_worker.quit_slot()

        self.io_thread.quit()
        self.io_thread.wait()
        self.infer_thread.quit()
        self.infer_thread.wait()
        super().closeEvent(event)

    def __onImageFolderCardClicked(self):
        """ download folder card clicked slot """
        file, _ =  QFileDialog.getOpenFileName(self, "Choose", "./")
        # folder = QFileDialog.getExistingDirectory(
        #     self, "Choose folder", "./")
        if not file or cfg.get(cfg.downloadFolder) == file:
            return
        print(file)
        cfg.set(cfg.det_image_or_dir, file)
        self.imageFolderCard.setContent(file)
