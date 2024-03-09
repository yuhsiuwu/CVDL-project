import sys
from PyQt5 import QtWidgets
from controller import MainWindowController
from VGG19 import VGG19

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindowController()
    window.show()
    sys.exit(app.exec_())