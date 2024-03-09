from multiprocessing.connection import wait
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
from PyQt5.QtCore import QTimer
from vgg19_classifier import Ui_MainWindow
from matplotlib import pyplot as plt

from VGG19 import VGG19
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchsummary import summary

import time
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F
import cv2 as cv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MainWindowController(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.objpoints = []
        self.imgpoints = []
        self.images_Path = {}
        self.image_extrinsic = {}
        self.image_intrinsic = None
        self.image_distortion = None
        self.image_shape = None
        self.imageR_Path = None
        self.imageL_Path = None
        self.imgeR_draw = None
        self.word_anchor = {
            'word_1': 62,
            'word_2': 59,
            'word_3': 56,
            'word_4': 29,
            'word_5': 26,
            'word_6': 23
        }
        self.image_1_Path = None
        self.image_2_Path = None
        self.ratio = 0.50
        self.image_path = ""
        self.batch_size = 64
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.train_dataset = datasets.CIFAR10(root='data', train=True, transform=transforms.ToTensor(), download=True)
        self.model_path = "C:\\Users\\liyin.bai\\Desktop\\cvdl_hw1\\data\\model_record.pt"
        self.image_accuracy_and_loss = "C:\\Users\\liyin.bai\\Desktop\\cvdl_hw1\\data\\accuracy_and_loss.png"

    def setup_control(self):
        self.ui.load_folder_button.clicked.connect(self.read_folder)
        self.ui.load_imageL_button.clicked.connect(self.read_imageL)
        self.ui.load_imageR_button.clicked.connect(self.read_imageR)
        self.ui.find_corners_button.clicked.connect(self.find_corner)
        self.ui.find_intrinsic_button.clicked.connect(self.find_intrinsic)
        self.ui.find_extrinsic_button.clicked.connect(self.find_extrinsic)
        self.ui.find_distortion_button.clicked.connect(self.find_distortion)
        self.ui.show_result_button.clicked.connect(self.show_result)
        self.ui.show_words_on_board_button.clicked.connect(self.show_words_on_board)
        self.ui.show_words_vertically_button.clicked.connect(self.show_words_on_vertical)
        self.ui.stereo_disparity_map_button.clicked.connect(self.stereo_disparity)
        self.ui.find_extrinsic_combobox.addItems([str(n) for n in range(1, 16)])
        self.ui.load_image_1_button.clicked.connect(self.read_image_1)
        self.ui.load_image_2_button.clicked.connect(self.read_image_2)
        self.ui.keypoints_button.clicked.connect(self.find_keypoints)
        self.ui.matched_keypoints_button.clicked.connect(self.matched_keypoints) 
        self.ui.load_image_button.clicked.connect(self.load_image)
        self.ui.show_model_structure_button.clicked.connect(self.show_model_structure)
        self.ui.show_data_augmentation_button.clicked.connect(self.show_data_augmentation)
        self.ui.show_accuracy_and_loss_button.clicked.connect(self.show_accuracy_and_loss)
        self.ui.inference_button.clicked.connect(self.inference)

    def read_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./")
        if folder_path == '':
            print("請選擇路徑")
        else:
            dirs = os.listdir(folder_path)
            for file in dirs:
                if file.endswith('.bmp'):
                    filenum = file.split('.')[0]
                    self.images_Path[filenum] = folder_path + "/" + file


    def read_imageL(self):
        self.imageL_Path, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")

    def read_imageR(self):
        self.imageR_Path, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
    def folder_error_detect(self):
        if len(self.images_Path) == 0:
            print("請先選擇路徑")
            return False
        else:
            return True

    def file_error_detect(self):
        if (self.imageL_Path == None or self.imageR_Path == None):
            print("請先選擇路徑")
            return False
        else:
            return True

    def find_corner(self):
        if(self.folder_error_detect()):
            pattern_rows = 11
            pattern_cols = 8
            pattern_size = (pattern_rows, pattern_cols)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            object_points = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
            object_points[:, :2] = np.mgrid[0:pattern_rows, 0:pattern_cols].T.reshape(-1, 2)
            objpoints = []
            imgpoints = []
            for item in list(self.images_Path.items()):
                filePath = item[1]
                filename = item[0]
                image = cv.imread(filePath)
                gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                image_shape = gray_image.shape[::-1]
                ret, corners = cv.findChessboardCorners(gray_image, (pattern_rows, pattern_cols), None)
                if ret == True:
                    objpoints.append(object_points)
                    corners2 = cv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)
                    cv.drawChessboardCorners(image, (pattern_rows, pattern_cols), corners2, ret)
                    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
                    self.image_extrinsic[filename] = self.get_intrinsic(rvecs, tvecs)
                    image = cv.resize(image, (720, 480))
                    cv.imshow('img', image)
                    cv.waitKey(500)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
            self.image_intrinsic = mtx
            self.image_distortion = dist
            cv.destroyAllWindows()

    def get_intrinsic(self, rvecs, tvecs):
        rotationMatrix = np.zeros((3, 3), np.float64)
        cv.Rodrigues(rvecs[0], rotationMatrix)
        tvecs = np.array(tvecs[0])
        extirinsic = np.concatenate((rotationMatrix, tvecs), axis=1)
        return extirinsic

    def image_intrinsic_error_detect(self):
        if(self.image_intrinsic is None):
            print("請先計算內外參")
            return False
        else:
            return True

    def find_intrinsic(self):
        if(self.image_intrinsic_error_detect()):
            print(self.image_intrinsic)

    def find_extrinsic(self):
        if(self.image_intrinsic_error_detect()):
            index = self.ui.find_extrinsic_combobox.currentText()
            print(self.image_extrinsic.get(index))

    def find_distortion(self):
        if(self.image_intrinsic_error_detect()):
            print(self.image_distortion)

    def show_result(self):
        if(self.image_intrinsic_error_detect()):
            for item in list(self.images_Path.items()):
                filePath = item[1]
                filename = item[0]
                img = cv.imread(filePath)
                h, w = img.shape[:2]
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.image_intrinsic, self.image_distortion, (w,h), 1, (w,h))
                mapx, mapy = cv.initUndistortRectifyMap(self.image_intrinsic, self.image_distortion, None, newcameramtx, (w, h), 5)
                dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                image = cv.resize(img, (720, 480))
                image_dst = cv.resize(dst, (720, 480))
                cv.putText(image, 'distorted image', (10, 40), cv.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1, cv.LINE_AA)
                cv.putText(image_dst, 'undistorted image', (10, 40), cv.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1, cv.LINE_AA)
                vis = np.concatenate((image, image_dst), axis=1)
                cv.imshow('img', vis)
                cv.waitKey(1000)
            cv.destroyAllWindows()

    def show_words_on_board(self):
        if(self.folder_error_detect() and self.image_intrinsic_error_detect()):
            fs = cv.FileStorage('C:\\Users\\liyin.bai\\Desktop\\Dataset_CvDl_Hw1\\Q2_Image\\Q2_lib\\alphabet_lib_onboard.txt', cv.FILE_STORAGE_READ)
            pattern_rows = 11
            pattern_cols = 8
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            object_points = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
            object_points[:, :2] = np.mgrid[0:pattern_rows, 0:pattern_cols].T.reshape(-1, 2)
            text = self.ui.augmented_reality_lineedit.text()
            print (text)
            wordList = [str(n) for n in text]

            for item in list(self.images_Path.items()):
                fileName, filePath = item[0], item[1]
                img = cv.imread(filePath)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, (pattern_rows, pattern_cols), None)
                if ret == True:
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    ret, rvecs, tvecs = cv.solvePnP(object_points, corners2, self.image_intrinsic, self.image_distortion)
                    for index, word in enumerate(wordList, start=1):
                        anchor = object_points[self.word_anchor["word_{}".format(index)]]
                        ch = fs.getNode(word).mat()
                        ch = (ch[:, :, :] + anchor).reshape(-1, 3)
                        ch_point, jac = cv.projectPoints(ch, rvecs, tvecs, self.image_intrinsic, self.image_distortion)
                        iter_point = iter(ch_point)
                        for point_1 in iter_point:
                            point_2 = next(iter_point)
                            img = self.draw_line(img, point_1.astype(int), point_2.astype(int))
                    img = cv.resize(img, (720, 480))
                    cv.imshow('img', img)
                    cv.waitKey(1000)
                cv.destroyAllWindows()

    def show_words_on_vertical(self, object_points):
        if (self.folder_error_detect() and self.image_intrinsic_error_detect()):
            fs = cv.FileStorage('C:\\Users\\liyin.bai\\Desktop\\Dataset_CvDl_Hw1\\Q2_Image\\Q2_lib\\alphabet_lib_vertical.txt', cv.FILE_STORAGE_READ)
            pattern_rows = 11
            pattern_cols = 8
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            object_points = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
            object_points[:, :2] = np.mgrid[0:pattern_rows, 0:pattern_cols].T.reshape(-1, 2)
            text = self.ui.augmented_reality_lineedit.text()
            wordList = [str(n) for n in text]

            for item in list(self.images_Path.items()):
                fileName, filePath = item[0], item[1]
                img = cv.imread(filePath)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, (pattern_rows, pattern_cols), None)
                if ret == True:
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    ret, rvecs, tvecs = cv.solvePnP(object_points, corners2, self.image_intrinsic, self.image_distortion)
                    for index, word in enumerate(wordList, start=1):
                        anchor = object_points[self.word_anchor["word_{}".format(index)]]
                        ch = fs.getNode(word).mat()
                        ch = (ch[:, :, :] + anchor).reshape(-1, 3)
                        ch_point, jac = cv.projectPoints(ch, rvecs, tvecs, self.image_intrinsic, self.image_distortion)
                        iter_point = iter(ch_point)
                        for point_1 in iter_point:
                            point_2 = next(iter_point)
                            img = self.draw_line(img, point_1.astype(int), point_2.astype(int))
                    img = cv.resize(img, (720, 480))
                    cv.imshow('img', img)
                    cv.waitKey(1000)
                cv.destroyAllWindows()

    def draw_line(self, img, p1, p2):
        img = cv.line(img, tuple(p1.ravel()), tuple(p2.ravel()), (0, 0, 255), 5)
        return img

    def stereo_disparity(self):
        if(self.file_error_detect()):
            self.stereo_disparity_map()
            self.show_stereo_correspondence()

    def stereo_disparity_map(self):
        imgL = cv.imread(self.imageL_Path, 0)
        imgR = cv.imread(self.imageR_Path, 0)

        stereo = cv.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL, imgR)
        disparity = cv.normalize(disparity, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        disparity = cv.resize(disparity, (720, 480))

        cv.imshow('disparity', disparity)
        cv.waitKey(3000)
        cv.destroyAllWindows()

    def show_stereo_correspondence(self):
        imgL = cv.imread(self.imageL_Path, cv.IMREAD_UNCHANGED)
        imgR = cv.imread(self.imageR_Path, cv.IMREAD_UNCHANGED)

        imgL_gray = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        imgR_gray = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        stereo = cv.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL_gray, imgR_gray)
        disparity = cv.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)

        imgHeight, imgWidth  = imgL.shape[0], imgL.shape[1]
        imgL = cv.resize(imgL, (720, 480))
        imgR = cv.resize(imgR, (720, 480))
        cv.imshow('imgL', imgL)
        cv.imshow('imgR', imgR)

        def show_imgR_coordinate(event, x, y, flags, userdata):
            if event == cv.EVENT_LBUTTONDOWN:
                originX = int(x * (imgWidth/720))
                originY = int(y * (imgHeight/480))
                if(disparity[originY, originX] != 0):
                    imgR_originX = originX - disparity[originY, originX]
                    imgR_originY = originY
                    cv.circle(imgR,
                              (int(imgR_originX * (720/imgWidth)),
                               int(imgR_originY * (480/imgHeight))),
                               2, (0, 0, 255), -1)
                else:
                    print("資料缺失")
                cv.imshow('imgR', imgR)
        cv.setMouseCallback('imgL', show_imgR_coordinate)

    def read_image_1(self):
        self.image_1_Path, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")

    def read_image_2(self):
        self.image_2_Path, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")


    def find_keypoints(self):
        image = cv.imread(self.image_1_Path)
        image = cv.resize(image, (720, 480))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        detector = cv.xfeatures2d.SIFT_create()
        keypoints, destination = detector.detectAndCompute(gray, None)
        image_point = cv.drawKeypoints(image, keypoints, None)
        cv.imshow('img', image_point)
        cv.waitKey(5000)
        cv.destroyAllWindows()

    def matched_keypoints(self):
        image_1 = cv.imread(self.image_1_Path)
        image_2 = cv.imread(self.image_2_Path)
        gray_1 = cv.cvtColor(image_1, cv.COLOR_BGR2GRAY)
        gray_2 = cv.cvtColor(image_2, cv.COLOR_BGR2GRAY)

        detector = cv.xfeatures2d.SIFT_create()
        keypoints_1, dest_1 = detector.detectAndCompute(gray_1, None)
        keypoints_2, dest_2 = detector.detectAndCompute(gray_2, None)

        matcher = cv.BFMatcher()
        raw_mathces = matcher.knnMatch(dest_1, dest_2, k=2)
        good_matches = []
        for m1, m2 in raw_mathces:
            if m1.distance < self.ratio * m2.distance:
                good_matches.append([m1])

        matches = cv.drawMatchesKnn(
                        image_1, keypoints_1,
                        image_2, keypoints_2,
                        good_matches, None, flags=2)

        matches = cv.resize(matches, (720, 480))
        cv.imshow('img', matches)
        cv.waitKey(5000)
        cv.destroyAllWindows()


    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.loading_image()

    def loading_image(self):
        if os.path.isfile(self.image_path):
            width = self.ui.image_graphicsView.width()
            height = self.ui.image_graphicsView.height() 
            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QtGui.QPixmap(self.image_path).scaled(width, height)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.ui.image_graphicsView.setScene(scene)

    def show_train_image(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        for features, targets in train_loader:
            break
        features = features[:9]
        fig = plt.figure()
        for i in range(features.shape[0]):
            plt.subplot(3, 3, i+1)
            tmp = features[i]
            plt.axis('off')
            plt.imshow(np.transpose(tmp, (1, 2, 0)))
            plt.title(self.classes[targets[i]])

        plt.show()

    def show_model_structure(self):
        model = VGG19(num_classes=10)
        model = model.to(DEVICE)
        summary(model, (3, 32, 32))
        self.ui.inference_label.setText("finish")

    def show_data_augmentation(self):
        transform_Rotation = transforms.Compose([
            transforms.RandomRotation(30)
        ])
        transform_ResizedCrop = transforms.Compose([
            transforms.RandomResizedCrop((100, 200))
        ])
        transform_HorizontalFlip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.9)
        ])
        
        fileNameList = []
        folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./")
        dirs = os.listdir(folder_path)
        for file in dirs:
            if file.endswith('.png'):
                fileNameList.append(file)        
        
        img_list = []
        for i, image in enumerate(fileNameList):
            if (i % 3 == 0):
                img = Image.open(folder_path + "/" + image, mode='r')
                img_list.append(transform_Rotation(img))
            if (i % 3 == 1):
                img = Image.open(folder_path + "/" + image, mode='r')
                img_list.append(transform_ResizedCrop(img))
            if (i % 3 == 2):
                img = Image.open(folder_path + "/" + image, mode='r')
                img_list.append(transform_HorizontalFlip(img))
        fig = plt.figure()

        for index, img in enumerate(img_list):
            plt.subplot(3, 3, index+1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()
        

    def show_accuracy_and_loss(self):
        if os.path.isfile(self.image_accuracy_and_loss):
            width = self.ui.image_graphicsView.width()
            height = self.ui.image_graphicsView.height() 
            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QtGui.QPixmap(self.image_accuracy_and_loss).scaled(width, height)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.ui.image_graphicsView.setScene(scene)

    def run_inference(self):       
        image = Image.open(self.image_path).convert('RGB')
        transforms_ToTensor = transforms.Compose([
            transforms.ToTensor()
        ])
        image_resize = image.resize((32, 32))
        input_tensor = transforms_ToTensor(image_resize)
        input_tensor = input_tensor.unsqueeze(0)        
      
        self.model = torch.load(self.model_path)
        self.model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        confidence, predictions = self.model.forward(input_tensor.to(device))
        print(confidence)
        predicted_class = torch.argmax(predictions, dim=1)
        max_prob, predicted_class = torch.max(F.softmax(confidence, dim=1), dim=1) 
        print("max_prob = ", max_prob.item())
        print("label = ", predicted_class)

        QTimer.singleShot(0, lambda: self.inference_label.setText("Predicted: {}".format(self.classes[predicted_class])))    

        return confidence
    
    def inference(self):
        confidence = self.run_inference()       
        self.show_histogram(confidence)

    def show_histogram(self, confidence):
        probas = F.softmax(confidence, dim=1)
        confidence_values = probas.squeeze().cpu().detach().numpy()

        plt.bar(self.classes, confidence_values)
        plt.title('Probability of each class')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.show()






