from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(917, 721)
        MainWindow.setToolTipDuration(0)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.load_image_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.load_image_groupbox.setGeometry(QtCore.QRect(20, 20, 181, 311))
        self.load_image_groupbox.setObjectName("load_image_groupbox")
        self.load_folder_button = QtWidgets.QPushButton(self.load_image_groupbox)
        self.load_folder_button.setGeometry(QtCore.QRect(30, 50, 121, 41))
        self.load_folder_button.setObjectName("load_folder_button")
        self.load_imageL_button = QtWidgets.QPushButton(self.load_image_groupbox)
        self.load_imageL_button.setGeometry(QtCore.QRect(30, 130, 121, 41))
        self.load_imageL_button.setObjectName("load_imageL_button")
        self.load_imageR_button = QtWidgets.QPushButton(self.load_image_groupbox)
        self.load_imageR_button.setGeometry(QtCore.QRect(30, 210, 121, 41))
        self.load_imageR_button.setObjectName("load_imageR_button")
        self.augmented_reality_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.augmented_reality_groupbox.setGeometry(QtCore.QRect(450, 20, 201, 311))
        self.augmented_reality_groupbox.setObjectName("augmented_reality_groupbox")
        self.show_words_on_board_button = QtWidgets.QPushButton(self.augmented_reality_groupbox)
        self.show_words_on_board_button.setGeometry(QtCore.QRect(30, 140, 141, 31))
        self.show_words_on_board_button.setObjectName("show_words_on_board_button")
        self.show_words_vertically_button = QtWidgets.QPushButton(self.augmented_reality_groupbox)
        self.show_words_vertically_button.setGeometry(QtCore.QRect(30, 200, 141, 31))
        self.show_words_vertically_button.setObjectName("show_words_vertically_button")
        self.augmented_reality_lineedit = QtWidgets.QLineEdit(self.augmented_reality_groupbox)
        self.augmented_reality_lineedit.setGeometry(QtCore.QRect(20, 50, 161, 51))
        self.augmented_reality_lineedit.setObjectName("augmented_reality_lineedit")
        self.stereo_disparity_map_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.stereo_disparity_map_groupbox.setGeometry(QtCore.QRect(680, 20, 201, 311))
        self.stereo_disparity_map_groupbox.setObjectName("stereo_disparity_map_groupbox")
        self.stereo_disparity_map_button = QtWidgets.QPushButton(self.stereo_disparity_map_groupbox)
        self.stereo_disparity_map_button.setGeometry(QtCore.QRect(20, 130, 161, 31))
        self.stereo_disparity_map_button.setObjectName("stereo_disparity_map_button")
        self.VGG19_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.VGG19_groupbox.setGeometry(QtCore.QRect(450, 350, 171, 261))
        self.VGG19_groupbox.setObjectName("VGG19_groupbox")
        self.load_image_button = QtWidgets.QPushButton(self.VGG19_groupbox)
        self.load_image_button.setGeometry(QtCore.QRect(20, 30, 131, 23))
        self.load_image_button.setObjectName("load_image_button")
        self.show_data_augmentation_button = QtWidgets.QPushButton(self.VGG19_groupbox)
        self.show_data_augmentation_button.setGeometry(QtCore.QRect(20, 80, 131, 31))
        self.show_data_augmentation_button.setObjectName("show_data_augmentation_button")
        self.show_model_structure_button = QtWidgets.QPushButton(self.VGG19_groupbox)
        self.show_model_structure_button.setGeometry(QtCore.QRect(20, 130, 131, 23))
        self.show_model_structure_button.setObjectName("show_model_structure_button")
        self.show_accuracy_and_loss_button = QtWidgets.QPushButton(self.VGG19_groupbox)
        self.show_accuracy_and_loss_button.setGeometry(QtCore.QRect(20, 170, 131, 23))
        self.show_accuracy_and_loss_button.setObjectName("show_accuracy_and_loss_button")
        self.inference_button = QtWidgets.QPushButton(self.VGG19_groupbox)
        self.inference_button.setGeometry(QtCore.QRect(20, 220, 131, 23))
        self.inference_button.setObjectName("inference_button")
        self.calibration_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.calibration_groupbox.setGeometry(QtCore.QRect(230, 20, 191, 311))
        self.calibration_groupbox.setObjectName("calibration_groupbox")
        self.find_corners_button = QtWidgets.QPushButton(self.calibration_groupbox)
        self.find_corners_button.setGeometry(QtCore.QRect(40, 20, 101, 31))
        self.find_corners_button.setObjectName("find_corners_button")
        self.find_intrinsic_button = QtWidgets.QPushButton(self.calibration_groupbox)
        self.find_intrinsic_button.setGeometry(QtCore.QRect(40, 60, 101, 31))
        self.find_intrinsic_button.setObjectName("find_intrinsic_button")
        self.find_extrinsic_groupbox = QtWidgets.QGroupBox(self.calibration_groupbox)
        self.find_extrinsic_groupbox.setGeometry(QtCore.QRect(20, 100, 141, 111))
        self.find_extrinsic_groupbox.setObjectName("find_extrinsic_groupbox")
        self.find_extrinsic_button = QtWidgets.QPushButton(self.find_extrinsic_groupbox)
        self.find_extrinsic_button.setGeometry(QtCore.QRect(20, 60, 101, 31))
        self.find_extrinsic_button.setObjectName("find_extrinsic_button")
        self.find_extrinsic_combobox = QtWidgets.QComboBox(self.find_extrinsic_groupbox)
        self.find_extrinsic_combobox.setGeometry(QtCore.QRect(40, 30, 61, 22))
        self.find_extrinsic_combobox.setObjectName("find_extrinsic_combobox")
        self.find_distortion_button = QtWidgets.QPushButton(self.calibration_groupbox)
        self.find_distortion_button.setGeometry(QtCore.QRect(40, 220, 101, 31))
        self.find_distortion_button.setObjectName("find_distortion_button")
        self.show_result_button = QtWidgets.QPushButton(self.calibration_groupbox)
        self.show_result_button.setGeometry(QtCore.QRect(40, 260, 101, 31))
        self.show_result_button.setObjectName("show_result_button")
        self.SIFT_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.SIFT_groupbox.setGeometry(QtCore.QRect(230, 350, 191, 241))
        self.SIFT_groupbox.setObjectName("SIFT_groupbox")
        self.load_image_1_button = QtWidgets.QPushButton(self.SIFT_groupbox)
        self.load_image_1_button.setGeometry(QtCore.QRect(40, 30, 121, 21))
        self.load_image_1_button.setObjectName("load_image_1_button")
        self.load_image_2_button = QtWidgets.QPushButton(self.SIFT_groupbox)
        self.load_image_2_button.setGeometry(QtCore.QRect(40, 80, 121, 23))
        self.load_image_2_button.setObjectName("load_image_2_button")
        self.keypoints_button = QtWidgets.QPushButton(self.SIFT_groupbox)
        self.keypoints_button.setGeometry(QtCore.QRect(40, 130, 121, 23))
        self.keypoints_button.setObjectName("keypoints_button")
        self.matched_keypoints_button = QtWidgets.QPushButton(self.SIFT_groupbox)
        self.matched_keypoints_button.setGeometry(QtCore.QRect(40, 180, 121, 23))
        self.matched_keypoints_button.setObjectName("matched_keypoints_button")
        self.image_graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.image_graphicsView.setGeometry(QtCore.QRect(630, 390, 251, 311))
        self.image_graphicsView.setObjectName("image_graphicsView")
        self.inference_label = QtWidgets.QLabel(self.centralwidget)
        self.inference_label.setEnabled(True)
        self.inference_label.setGeometry(QtCore.QRect(640, 360, 231, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.inference_label.setFont(font)
        self.inference_label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.inference_label.setText("")
        self.inference_label.setAlignment(QtCore.Qt.AlignCenter)
        self.inference_label.setObjectName("inference_label")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.load_image_groupbox.setTitle(_translate("MainWindow", "Load Image"))
        self.load_folder_button.setText(_translate("MainWindow", "Load folder"))
        self.load_imageL_button.setText(_translate("MainWindow", "Load Image_L"))
        self.load_imageR_button.setText(_translate("MainWindow", "Load Image_R"))
        self.augmented_reality_groupbox.setTitle(_translate("MainWindow", "2. Augmented Reality"))
        self.show_words_on_board_button.setText(_translate("MainWindow", "2.1 show words on board"))
        self.show_words_vertically_button.setText(_translate("MainWindow", "2.2 show words vertical"))
        self.stereo_disparity_map_groupbox.setTitle(_translate("MainWindow", "3. Stereo disparity map"))
        self.stereo_disparity_map_button.setText(_translate("MainWindow", "3.1 stereo disparity map"))
        self.VGG19_groupbox.setTitle(_translate("MainWindow", "5. VGG19"))
        self.load_image_button.setText(_translate("MainWindow", "Load Image"))
        self.show_data_augmentation_button.setText(_translate("MainWindow", "5.1 Show Augmented\n"
"Images"))
        self.show_model_structure_button.setText(_translate("MainWindow", "5.2 Show Model Structure"))
        self.show_accuracy_and_loss_button.setText(_translate("MainWindow", "5.3 Show Acc and Loss"))
        self.inference_button.setText(_translate("MainWindow", "5.4 Inference"))
        self.calibration_groupbox.setTitle(_translate("MainWindow", "1. Calibration"))
        self.find_corners_button.setText(_translate("MainWindow", "1.1 Find corners"))
        self.find_intrinsic_button.setText(_translate("MainWindow", "1.2 Find intrinsic"))
        self.find_extrinsic_groupbox.setTitle(_translate("MainWindow", "1.3 Find extrinsic"))
        self.find_extrinsic_button.setText(_translate("MainWindow", "1.3 Find extrinsic"))
        self.find_distortion_button.setText(_translate("MainWindow", "1.4 Find distortion"))
        self.show_result_button.setText(_translate("MainWindow", "1.5 Show result"))
        self.SIFT_groupbox.setTitle(_translate("MainWindow", "4. SIFT"))
        self.load_image_1_button.setText(_translate("MainWindow", "Load Image1"))
        self.load_image_2_button.setText(_translate("MainWindow", "Load Image2"))
        self.keypoints_button.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.matched_keypoints_button.setText(_translate("MainWindow", "4.2 Matched Keypoints"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
