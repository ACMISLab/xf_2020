#coding:utf-8
from qfluentwidgets import qconfig, QConfig, ConfigItem, FolderValidator, OptionsConfigItem, OptionsValidator


class Config(QConfig):
    downloadFolder = ConfigItem(
        "Folders", "Download", "app/download", FolderValidator())

    det_image_or_dir = ConfigItem('files', 'images', './')

    model_dir = ConfigItem("files", 'model', './resource/model-onnx')

    defect_labels = OptionsConfigItem('models', 'labels', '浅色块',  OptionsValidator(['白点', '浅色块', '深色点块', '光圈', '划伤']), restart=True)

    record_sync_url = ConfigItem(group='record', name='url', default='localhost:8080')

cfg = Config()
qconfig.load('app/config/config.json', cfg)