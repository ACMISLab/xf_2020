# coding:utf-8
import json

import dateutil.utils
from PySide6.QtCore import QUrl, Signal, QObject, QFileInfo, QDateTime
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QHttpMultiPart, QHttpPart


def generate_multipart_data(text_dict=None, file=None):
    multipart_data = QHttpMultiPart(QHttpMultiPart.FormDataType, None)
    if text_dict:
        for key, value in text_dict.items():
            text_part = QHttpPart()
            text_part.setHeader(QNetworkRequest.ContentDispositionHeader, "form-data;name=\"%s\"" % key)
            text_part.setBody(str(value).encode("utf-8"))
            multipart_data.append(text_part)
    if file is None:
        file_part = QHttpPart()
        current_datetime = QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')
        file_part.setHeader(QNetworkRequest.ContentDispositionHeader,
                            "form-data; name=%s; filename=%s" % ('img', current_datetime + '.jpg'))
        file_part.setBodyDevice(file)
        file.setParent(multipart_data)
        multipart_data.append(file_part)

    return multipart_data


class NetworkManager(QObject):
    get_reply_finished = Signal(int, dict)  # 自定义信号，将接收到的数据发送至外部处理
    post_reply_finished = Signal(int, dict)
    progress_signal = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.manager = QNetworkAccessManager(self)
        self.reply = None

    def get(self, url, query_params=None):
        if query_params is not None:
            request = QNetworkRequest(QUrl(url + "?" + query_params))
        else:
            request = QNetworkRequest(QUrl(url))
        self.reply = self.manager.get(request)
        self.reply.finished.connect(self.getReply)  # 将reply的finished信号与槽函数连接

    def post(self, url, form_data: dict = None, file=None):
        if self.reply is not None:
            self.reply.disconnect(self)
        request = QNetworkRequest(QUrl(url))
        multipart_data = generate_multipart_data(form_data, file)

        '''
        传入dict类型的FormData和FileData，返回QHttpMultiPart对象
		'''
        self.reply = self.manager.post(request, multipart_data)
        multipart_data.setParent(self.reply)
        '''
        post时必须要为multipart 对象设置parent 否则程序将直接退出不返回任何数据
        '''
        self.reply.finished.connect(self.postReply)
        self.reply.errorOccurred.connect(self.handleError)
        self.reply.uploadProgress.connect(self.setProgress)

    def getReply(self):
        bytes_string = self.reply.readAll()
        if len(bytes_string) > 0:
            received_data = json.loads(str(bytes_string, "utf-8"))
            print(received_data)
        else:
            received_data = {}
        self.get_reply_finished.emit(200, received_data)

    def postReply(self):
        bytes_string = self.reply.readAll()
        if len(bytes_string) > 0:
            received_data = json.loads(str(bytes_string, "utf-8"))
        else:
            received_data = {}
        self.post_reply_finished.emit(200, received_data)

    def setProgress(self, bytes_sent, bytes_total):
        self.progress_signal.emit(bytes_sent, bytes_total)

    def handleError(self):
        self.get_reply_finished.emit(400, {'error': self.reply.errorString()})
