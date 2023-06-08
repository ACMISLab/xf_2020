# coding:utf-8
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QLabel, QApplication, QHBoxLayout, QStackedWidget, QFrame
from qfluentwidgets import NavigationInterface, NavigationItemPosition

from qfluentwidgets import FluentIcon as FIF

from qframelesswindow import FramelessWindow, TitleBar, StandardTitleBar

from app.views import DetectWidget, InferenceConfig


class CustomTitleBar(TitleBar):
    """ Title bar with icon and title """

    def __init__(self, parent):
        super().__init__(parent)
        # add window icon
        self.iconLabel = QLabel(self)
        self.iconLabel.setFixedSize(18, 18)
        self.hBoxLayout.insertSpacing(0, 10)
        self.hBoxLayout.insertWidget(1, self.iconLabel, 0, Qt.AlignLeft | Qt.AlignBottom)
        self.window().windowIconChanged.connect(self.setIcon)

        # add title label
        self.titleLabel = QLabel(self)
        self.hBoxLayout.insertWidget(2, self.titleLabel, 0, Qt.AlignLeft | Qt.AlignBottom)
        self.titleLabel.setObjectName('titleLabel')
        self.window().windowTitleChanged.connect(self.setTitle)

    def setTitle(self, title):
        self.titleLabel.setText(title)
        self.titleLabel.adjustSize()

    def setIcon(self, icon):
        self.iconLabel.setPixmap(QIcon(icon).pixmap(18, 18))


class Widget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.label = QLabel(text, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)

        # leave some space for title bar
        self.hBoxLayout.setContentsMargins(0, 32, 0, 0)


class Window(FramelessWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitleBar(StandardTitleBar(self))

        self.hBoxLayout = QHBoxLayout(self)
        self.navigationInterface = NavigationInterface(
            self, showMenuButton=True, showReturnButton=True)
        self.stackWidget = QStackedWidget(self)

        # create sub interface
        self.detectInterface = DetectWidget('检测', self)
        self.inferenceConfigInterface = InferenceConfig('配置', self)
        self.videoInterface = Widget('Video Interface', self)
        self.folderInterface = Widget('Folder Interface', self)
        self.settingInterface = Widget('Setting Interface', self)

        self.stackWidget.addWidget(self.detectInterface)
        self.stackWidget.addWidget(self.inferenceConfigInterface)
        self.stackWidget.addWidget(self.videoInterface)
        self.stackWidget.addWidget(self.folderInterface)
        self.stackWidget.addWidget(self.settingInterface)

        # initialize layout
        self.initLayout()

        # add items to navigation interface
        self.initNavigation()

        self.initWindow()

    def initLayout(self):
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addWidget(self.stackWidget)
        self.hBoxLayout.setStretchFactor(self.stackWidget, 0)

        self.titleBar.raise_()
        self.navigationInterface.displayModeChanged.connect(self.titleBar.raise_)

    def switchTo(self, widget):
        self.stackWidget.setCurrentWidget(widget)

    def onCurrentInterfaceChanged(self, index):
        widget = self.stackWidget.widget(index)
        self.navigationInterface.setCurrentItem(widget.objectName())

    def initNavigation(self):

        self.navigationInterface.addItem(
            routeKey=self.detectInterface.objectName(),
            icon=FIF.PHOTO,
            text='缺陷检测',
            onClick=lambda: self.switchTo(self.detectInterface)
        )

        self.navigationInterface.addItem(
            routeKey=self.inferenceConfigInterface.objectName(),
            icon=FIF.FOLDER,
            text='检测配置',
            onClick=lambda: self.switchTo(self.inferenceConfigInterface)
        )

        self.navigationInterface.addItem(
            routeKey=self.videoInterface.objectName(),
            icon=FIF.VIDEO,
            text='Video library',
            onClick=lambda: self.switchTo(self.videoInterface)
        )

        self.navigationInterface.addSeparator()

        # add navigation items to scroll area
        self.navigationInterface.addItem(
            routeKey=self.folderInterface.objectName(),
            icon=FIF.FOLDER,
            text='Folder library',
            onClick=lambda: self.switchTo(self.folderInterface),
            position=NavigationItemPosition.SCROLL
        )


        self.navigationInterface.addItem(
            routeKey=self.settingInterface.objectName(),
            icon=FIF.SETTING,
            text='Settings',
            onClick=lambda: self.switchTo(self.settingInterface),
            position=NavigationItemPosition.BOTTOM
        )

        # !IMPORTANT: don't forget to set the default route key
        self.navigationInterface.setDefaultRouteKey(self.inferenceConfigInterface.objectName())

        # set the maximum width
        # self.navigationInterface.setExpandWidth(300)

        self.stackWidget.currentChanged.connect(self.onCurrentInterfaceChanged)
        self.stackWidget.setCurrentIndex(0)

    def initWindow(self):

        self.resize(600, 500)
        self.setWindowIcon(QIcon('resource/logo.png'))
        self.setWindowTitle('Tile Defects detection app')
        self.titleBar.setAttribute(Qt.WA_StyledBackground)

        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.titleBar.raise_()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = Window()
    w.show()
    app.exec()
