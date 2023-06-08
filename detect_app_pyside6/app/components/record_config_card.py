from typing import Union

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from qfluentwidgets import ExpandLayout, FluentStyleSheet, SettingCard, OptionsConfigItem, FluentIconBase, \
    PlainTextEdit, PushButton, MessageBox

from app.components.network_manager import NetworkManager
from app.config.config import cfg


class RecordConfigCard(SettingCard):
    def __init__(self, configItem: OptionsConfigItem, icon: Union[str, QIcon, FluentIconBase], title, content=None,
                 parent=None):
        super().__init__(icon, title, content, parent)
        self.manager = NetworkManager()

        self.configItem = configItem

        self.protocolLabel = QLabel('http://', self)
        self.urlEdit = PlainTextEdit(self)
        # self.ossEdit = PlainTextEdit(self)

        self.urlTestButton = PushButton('测试并保存', self)

        self.hBoxLayout.addWidget(self.protocolLabel, 1, Qt.AlignRight)
        self.hBoxLayout.addWidget(self.urlEdit, 3, Qt.AlignRight)
        self.hBoxLayout.addWidget(self.urlTestButton, 1, Qt.AlignRight)

        self.hBoxLayout.addSpacing(16)

        self.urlEdit.setPlainText(self.configItem.value)

        self.urlOption = configItem.value
        self.urlTestButton.clicked.connect(self._onUrlTesting)

    def _onUrlTesting(self):
        if len(self.urlEdit.toPlainText().strip()) < 5:
            return
        url = self.urlEdit.toPlainText().strip()
        # self.urlEdit.clear()
        self.manager.get("http://{}/jeecg-boot/detect/record/echo".format(self.urlEdit.toPlainText()))
        self.manager.get_reply_finished.connect(
            lambda code, res : self.setValue(self.urlEdit.toPlainText().strip())if len(res.keys()) > 0 else {})




    def setValue(self, value):
        # self.urlEdit.setPlainText(value)

        cfg.set(self.configItem, value)
