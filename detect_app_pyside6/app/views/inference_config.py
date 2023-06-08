from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFileDialog

from qfluentwidgets import FluentIcon as FIF, InfoBar, ScrollArea, ComboBoxSettingCard
from qfluentwidgets import ExpandLayout, SettingCardGroup, PushSettingCard

from app.components.record_config_card import RecordConfigCard
from app.config.config import cfg


class InferenceConfig(QWidget):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.expandLayout = ExpandLayout(self)

        # music folders
        self.modelGroup = SettingCardGroup(
            "模型配置", self)

        self.modelFolderCard = PushSettingCard(
            '重新选择',
            FIF.DOWNLOAD,
            "Onnx Path",
            cfg.get(cfg.model_dir),
            self.modelGroup
        )

        self.labelGroup = SettingCardGroup(
            "类别标签", self)

        self.labelComboBoxCard = ComboBoxSettingCard(
            cfg.defect_labels,
            FIF.TAG,
            '缺陷类名',
            '按顺序编辑',
            texts=['白点', '浅色块', '深色点块', '光圈', '划伤'],
            parent=self.labelGroup
        )

        self.recordGroup = SettingCardGroup("检测数据同步", self)
        self.recordConfigCard = RecordConfigCard(
            cfg.record_sync_url,
            icon=FIF.UPDATE,
            title='同步网关',
            content='IP:端口',
            parent=self.recordGroup
        )

        self.__init_widget()

    def __init_widget(self):
        self.__initLayout()
        self.__connectSignalToSlot()

    def __initLayout(self):
        self.modelGroup.addSettingCard(self.modelFolderCard)
        self.labelGroup.addSettingCard(self.labelComboBoxCard)
        self.recordGroup.addSettingCard(self.recordConfigCard)

        self.expandLayout.setSpacing(20)
        self.expandLayout.setContentsMargins(36, 40, 36, 0)

        self.expandLayout.addWidget(self.modelGroup)
        self.expandLayout.addWidget(self.labelGroup)
        self.expandLayout.addWidget(self.recordGroup)

        self.setLayout(self.expandLayout)

    def __connectSignalToSlot(self):
        self.modelFolderCard.clicked.connect(self.__onModelFolderCardClicked)

    def __onModelFolderCardClicked(self):
        """ download folder card clicked slot """
        folder = QFileDialog.getExistingDirectory(
            self, "Choose folder", "./")
        if not folder or cfg.get(cfg.model_dir) == folder:
            return

        cfg.set(cfg.model_dir, folder)
        self.modelFolderCard.setContent(folder)
