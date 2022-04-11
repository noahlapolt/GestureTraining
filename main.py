import sys
import cv2
import time
import mediapipe as mp
from pynput.keyboard import Key, Controller, Listener
import math as m
import json
import os
import ctypes
import numpy as np
import torch

from PyQt5.QtCore import QThread, Qt, QObject, QEvent, QTimer, QRect, pyqtSignal
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage
import models
import mouse

# Noah's TODOs:
# TODO: Prevent empty binds.
# TODO: Add type once or repeat options.
# TODO: Add default bind.

# Craig's TODOs:
# TODO: Improve mouse smoothing

w = ctypes.windll.user32.GetSystemMetrics(78)
h = ctypes.windll.user32.GetSystemMetrics(79)
hidden_size = 1000

class CalibrationUI(QMainWindow):
    '''
    Brings together the menu and the calibration images.
    '''
    def __init__(self):
        super().__init__()

        # Loads in the save.
        if not os.path.exists('data.json'):
            with open("data.json", "w+") as f:
                f.write('{}')
                
        # Updates key tree if there is data.
        with open("data.json", "r") as f:
            self.ele_tree = json.load(f)

        # Loads the model.
        if len(self.ele_tree.keys()) > 0:
            self.model = models.FFN(
                len(self.ele_tree[list(self.ele_tree.keys())[0]][0]),
                hidden_size,
                len(self.ele_tree.keys()))
        else:
            self.model = None
        self.predict = self.model is not None
        self.train = False

        # Keyboard controler.
        self.key_ctrl = Controller()
        self.get_key = False
        self.get_macro = False

        # Window settings.
        self.setWindowTitle("Hand Mouse Calibration")
        self.resize(1280, 720)
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray_quit = False
        
        # Builds the main screen structrure.
        main = QWidget()
        layout = QHBoxLayout()
        sub_widget = QWidget()
        sub_lay = QVBoxLayout()
        self.tree_area = TreeArea(self)
        self.edit_area = EditArea(self)
        self.label = QLabel()
        self.label.setStyleSheet("background-color: #000000")
        self.label.setMinimumSize(100, 100)
        sub_lay.addWidget(self.tree_area,stretch=1)
        sub_lay.addWidget(self.edit_area,stretch=1)
        sub_widget.setLayout(sub_lay)
        sub_widget.setMinimumSize(100, 100)
        spliter = QSplitter(Qt.Horizontal)
        spliter.addWidget(sub_widget)
        spliter.addWidget(self.label)
        spliter.setSizes([100,200])
        layout.addWidget(spliter)
        main.setLayout(layout)

        # Thread for content to update on.
        self.progress = QProgressBar()
        self.thread = QThread()
        self.content = Content(self)
        self.content.moveToThread(self.thread)
        self.thread.started.connect(self.content.update_loop)
        self.content.progress.connect(lambda p: self.progress.setValue(p))

        # Tray setup
        tray_menu = QMenu(self)
        cali_action = QAction('Calibrate', self)
        cali_action.triggered.connect(self.view)
        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(self.quit)
        tray_menu.addAction(cali_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)
        self.tray.setContextMenu(tray_menu)
        self.tray.show()

        # Top of menu.
        menu = QMenuBar(self)
        file_menu = QMenu('&File', menu)
        menu.addMenu(file_menu)
        add_menu = QMenu('&Add', menu)
        file_menu.addMenu(add_menu)
        save_file = QAction('&Save', menu)
        save_file.triggered.connect(self.save)
        file_menu.addAction(save_file)
        quit_opt = QAction('&Quit', menu)
        quit_opt.triggered.connect(self.quit)
        file_menu.addSeparator()
        file_menu.addAction(quit_opt)

        # Set sub menu
        add_mouse = QMenu('&Mouse', add_menu)
        add_menu.addMenu(add_mouse)
        add_button = QAction('&Key',add_menu)
        add_button.triggered.connect(lambda: self.edit_area.add('Key'))
        add_menu.addAction(add_button)
        add_macro = QAction('&Macro',add_menu)
        add_macro.triggered.connect(lambda: self.edit_area.add('Macro'))
        add_menu.addAction(add_macro)

        # Mouse sub menu
        add_tracking = QAction('&Tracking', add_mouse)
        add_tracking.triggered.connect(lambda: self.edit_area.add('MOUSEMOVE'))
        add_mouse.addAction(add_tracking)
        add_left = QAction('&Left Button', add_mouse)
        add_left.triggered.connect(lambda: self.edit_area.add('MOUSELEFT'))
        add_mouse.addAction(add_left)
        add_right = QAction('&Right Button', add_mouse)
        add_right.triggered.connect(lambda: self.edit_area.add('MOUSERIGHT'))
        add_mouse.addAction(add_right)
        add_scroll = QAction('&Scroll', add_mouse)
        add_scroll.triggered.connect(lambda: self.edit_area.add('MOUSESCROLL'))
        add_mouse.addAction(add_scroll)

        # Default
        add_default = QAction('Default', add_menu)
        add_default.triggered.connect(lambda: self.edit_area.add('DEFAULT'))
        add_menu.addAction(add_default)

        # Setup window.
        self.setMenuBar(menu)
        self.setCentralWidget(main)
        self.thread.start()

    def view(self):
        '''
        Brings the calibration window back.
        '''
        self.show()

    def save(self):
        '''
        Saves all elements to file.
        '''
        # Saves data.
        with open("data.json", "w+") as f:
            to_save = json.dumps(self.ele_tree, indent=4, sort_keys=True)
            f.write(to_save)

        with open("data.json", "r") as f:
            self.ele_tree = json.load(f)

        # Updates options and title.
        self.setWindowTitle("Hand Mouse Calibration")
        self.train = len(self.ele_tree.keys()) > 0

    def quit(self):
        '''
        Quits the current program.
        '''
        self.tray_quit = True
        self.close()

    def closeEvent(self, event):
        if self.tray_quit:
            self.tray.hide()
            self.content.stop()
            self.thread.quit()
        else:
            event.ignore()
            self.hide()


class TreeArea(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.wid_lay = QVBoxLayout()

        # Builds the tree area.
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Current Settings'))
        self.scroll_area = QScrollArea()
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

        # Populates the key tree area.
        for ele in list(parent.ele_tree.keys()):
            if 'Macro' in ele:
                _type = 'Macro'
            elif 'MOUSE' in ele:
                _type = ele
            else:
                _type = 'Key'
            self.add(_type, ele)

    def add(self, _type, label):
        '''
        Adds a element to the scroll area so they are easy
        to edit and delete.

        Parameters
        ----------
        _type: str
            The type of element being added.
        label: str
            The label element to add.
        '''
        # Key setup
        new_ele = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))

        # Buttons
        edit_btn = QPushButton('Edit')
        edit_btn.clicked.connect(lambda: self.edit(new_ele, _type, label))
        del_btn = QPushButton('Delete')
        del_btn.clicked.connect(lambda: self.delete(new_ele, label))
        layout.addWidget(edit_btn)
        layout.addWidget(del_btn)

        # Adds new key
        widget = QWidget()
        new_ele.setLayout(layout)
        self.wid_lay.addWidget(new_ele)
        widget.setLayout(self.wid_lay)

        self.scroll_area.setWidget(widget)

    def edit(self, ele, _type, label):
        self.parent.setWindowTitle("Hand Mouse Calibration *")
        self.parent.edit_area.add(_type, label)
        self.delete(ele, label)

    def delete(self, ele, label):
        '''
        Deletes the element.

        Parameters
        ----------
        ele: QWidget
            The widget of the element that is going to be deleted.
        label: str
            The name of the element that is being deleted.
        '''
        self.parent.ele_tree.pop(label,None)
        ele.deleteLater()


class EditArea(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.target = QWidget(self)
        self.active = None

        # Builds the tree area.
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Editting'))
        self.scroll_area = QScrollArea()
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

        # Keyboard listener.
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

    def add(self, _type, label=''):
        if _type == 'Key':
            self.target = QWidget()
            layout = QVBoxLayout()
            edit = QLineEdit(label)
            edit.setReadOnly(True)
            layout.addWidget(edit)
            set_btn = QPushButton('Set')
            set_btn.clicked.connect(lambda: self.update(_type, self.active.text()))
            layout.addWidget(set_btn)
            self.target.setLayout(layout)
            self.active = edit
        elif _type == 'Macro':
            self.texts = []
            self.delays = []
            self.target = QWidget()
            self.wid_lay = QVBoxLayout()
            self.target.resize(self.scroll_area.width()-5, self.scroll_area.height()-5)
            layout = QVBoxLayout()
            self.target.setLayout(layout)
            self.scroll = QScrollArea()
            layout.addWidget(self.scroll)
            add_btn = QPushButton('Add')
            add_btn.clicked.connect(lambda: self.add_key())
            layout.addWidget(add_btn)
            set_btn = QPushButton('Set')
            set_btn.clicked.connect(self.set_macro)
            layout.addWidget(set_btn)

            # Add all from label
            if label != '':
                values = label.split(' ')[1:]
                for val in values:
                    cur = val.split(':')
                    self.add_key(cur[0], int(cur[1]))
        else:
            self.target = QWidget()
            layout = QVBoxLayout()
            edit = QLineEdit(_type)
            edit.setReadOnly(True)
            layout.addWidget(edit)
            set_btn = QPushButton('Set')
            set_btn.clicked.connect(lambda: self.update(_type, _type))
            layout.addWidget(set_btn)
            self.target.setLayout(layout)
            self.active = None

        self.parent.predict = False
        self.scroll_area.setWidget(self.target)

    def update(self, _type, label, i=0):
        '''
        Adds/Updates an element in the tree with its data.

        Parameters
        ----------
        label: str
            The label of the element to add/update.
        '''
        max_data = 10
        if label not in self.parent.ele_tree or len(self.parent.ele_tree[label]) < max_data or i < max_data:
            self.parent.predict = False
            
            if label not in self.parent.ele_tree:
                self.parent.ele_tree[label] = []

            data = self.parent.content.get_data()
            if len(data) > 0:
                self.parent.ele_tree[label].append(data)
                if i+1 != max_data:
                    print(f'Collecting data point {i+2} in 500ms.')
                else:
                    print('Completed!')
                QTimer.singleShot(500, lambda: self.update(_type, label, i+1))
            else:
                print(f'Collecting data point {i+1} in 500ms.')
                QTimer.singleShot(500, lambda: self.update(_type, label, i))
        else:
            self.parent.setWindowTitle('Hand Mouse Calibration *')
            self.active = None
            self.target.deleteLater()

            if len(self.parent.ele_tree.keys()) != self.parent.tree_area.wid_lay.count():
                self.parent.tree_area.add(_type, label)

            self.parent.predict = self.parent.model is not None

    def add_key(self, key='', delay=0):
        '''
        Class that adds a key to the macro element.
        '''
        # Main widget
        widget = QWidget()
        layout = QHBoxLayout()

        # Label
        edit = QLineEdit(key)
        edit.setReadOnly(True)
        layout.addWidget(edit,stretch=3)

        # Delay
        dlay = QSpinBox()
        dlay.setRange(0, 99999)
        dlay.setSuffix('ms')
        dlay.setValue(delay)
        layout.addWidget(dlay,stretch=2)
        
        # Buttons
        edit_btn = QPushButton('Edit')
        edit_btn.clicked.connect(lambda: self.set_active(edit))
        layout.addWidget(edit_btn, stretch=1)
        del_btn = QPushButton('Delete')
        index = self.wid_lay.count()
        del_btn.clicked.connect(lambda: self.del_macro(widget, index))
        layout.addWidget(del_btn, stretch=1)

        # Brings it all together.
        parent = QWidget()
        widget.setLayout(layout)
        self.wid_lay.addWidget(widget)
        parent.setLayout(self.wid_lay)
        self.scroll.setWidget(parent)

        # Keeps track of text and delays.
        self.texts.append(edit)
        self.delays.append(dlay)
        self.active = edit

    def set_macro(self):
        '''
        Sets the new macro in the tree.
        '''
        macro_str = 'Macro:'
        for i in range(len(self.texts)):
            macro_str += f' {self.texts[i].text()}:{self.delays[i].value()}'
        self.update('Macro', macro_str)

    def del_macro(self, target, index):
        self.texts.pop(index)
        self.delays.pop(index)
        self.active = None
        target.deleteLater()

    def set_active(self, target):
        '''
        Sets the active value to the target.

        Parameters
        ----------
        target: QWidget
            The widget that is now active.
        '''
        self.active = target

    def on_press(self, key):
        res = str(key).strip("\'").replace("Key.", "").lower()
        
        # Prevents missing '.
        if res == "":
            res = "\'"

        # Updates key.
        if self.active:
            self.active.setText(res)


class Content(QObject):
    progress = pyqtSignal(int)
    '''
    Displays the camera input for the calibration.
    '''
    def __init__(self, parent):
        '''
        Builds the Content for the screen.

        Parameters
        ----------
        parent: PyQt5Widget
            The Qt element right above this object.
        '''
        super().__init__()
        self.cam = cv2.VideoCapture(0)
        self.ht = HandTracking()
        self.parent = parent
        
        # Camera settings
        self.resolution = [self.cam.get(cv2.CAP_PROP_FRAME_WIDTH), self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)]

        # Determinds the fps of the camera if it could not be found.
        if self.cam.get(cv2.CAP_PROP_FPS) == 0:
            start_time = time.gmtime().tm_sec
            num_frames = 0
            
            # Times the fps.
            while ((start_time + 1) % 60 >= time.gmtime().tm_sec and time.gmtime().tm_sec != 59):
                self.cam.read()
                num_frames += 1
            
            # Sets the cam fps.
            self.cam.set(cv2.CAP_PROP_FPS, num_frames)

        # Updates the camera.
        self.running = self.cam.isOpened()
        self.last_key = None
        self.last_res = None
        self.mode = None

    def stop(self):
        '''
        Stops the video feed.
        '''
        self.running = False
        self.cam.release()
        cv2.destroyAllWindows()

    def get_data(self, lms=None):
        data = []
        if lms is None:
            lms = self.ht.find_landmarks()

        for i in range(len(lms)):
            if len(lms[i]) >= 21:
                # Collects data from joints.
                base = m.dist(lms[i][0], lms[i][5])
                angle = m.atan2(lms[i][0][1]-lms[i][5][1],lms[i][0][0]-lms[i][5][0])
                for point1 in lms[i]:
                    for point2 in lms[i]:
                        if point1 != point2:
                            data.append((m.dist(point1, point2)/base)*(angle-m.atan2(point1[1]-point2[1], point1[0]-point2[0])))

        # Returns empty array if not enough data.
        if len(data) < 420:
            return []

        # Corrects data shape.
        data.extend([0] * 420) if len(data) < 840 else data
        data = data[:840]
        return data

    def press_key(self, key):
        if key in dir(Key):
            key = Key[key]

        # Press the key if it is not pressed.
        if self.last_key != key:
            if key:
                self.parent.key_ctrl.press(key)
            if self.last_key:
                self.parent.key_ctrl.release(self.last_key)
            self.last_key = key
        
    def update_loop(self):
        '''
        Gets the current image and displays it in the content.
        '''
        while self.running:
            _, frame = self.cam.read()
            frame = cv2.flip(frame, 1)

            # Checking if we are able to detect the hand...
            img = self.ht.find_hands(frame)
            lms = self.ht.find_landmarks()
            ratio = self.resolution[0]/self.resolution[1]
            if self.parent.label.width() > 5:
                img = cv2.resize(img, (self.parent.label.width()-5, int((self.parent.label.width()-5)/ratio)), interpolation=cv2.INTER_AREA)

            # Gets image format
            if img.shape[2] == 4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888

            # Makes image into Qt image.
            qt_img = QImage(img.data,
                img.shape[1],
                img.shape[0], 
                img.strides[0],
                qformat)
            qt_img = qt_img.rgbSwapped()

            # Updates image.
            if not self.parent.isHidden():
                self.parent.label.setPixmap(QPixmap.fromImage(qt_img))
            
            if self.mode == 'MOVE':
                if len(lms) > 0:
                    pos = self.ht.get_pos(lms[0][0])
                    mouse.move(pos[0], pos[1], absolute=True, duration=0.01)

            if self.mode == 'SCROLL':
                if len(lms) > 0:
                    pos = self.ht.get_pos(lms[0][0])
                    mouse.wheel(-((pos[1]+(h/4))/(h/2) - 1.5))

            # Make the hand prediction.
            if self.parent.predict and not self.parent.train and len(self.parent.ele_tree.keys()) > 0 and len(lms)>0 and len(lms[0])>0:
                pred = models.predict_ffn(self.get_data(lms), self.parent.model)
                if pred[1] > 0.75:
                    res = list(self.parent.ele_tree.keys())[pred[0]]
                    if 'Macro' in res:
                        strokes = res.split(' ')[1:]
                        for stroke in strokes:
                            press = stroke.split(':')
                            self.press_key(press[0])
                            time.sleep(int(press[1]) / 1000)
                        self.press_key(None)
                    elif 'MOUSE' in res:
                        if res != self.last_res:
                            mouse.release('left')
                            if res == 'MOUSEMOVE':
                                if self.mode != 'MOVE':
                                    self.mode = 'MOVE'
                                print('Mode Switched')
                            elif res == 'MOUSELEFT':
                                mouse.press('left')
                            elif res == 'MOUSERIGHT':
                                mouse.click('right')
                            elif res =='MOUSESCROLL':
                                if self.mode == 'SCROLL':
                                    self.mode = None
                                else:
                                    self.mode = 'SCROLL'
                                print('Mode Switched')
                            self.last_res = res
                        self.press_key(None)
                    elif 'DEFAULT' in res:
                        if res != self.last_res:
                            self.mode = None
                            self.press_key(None)
                            self.last_res = 'DEFAULT'
                    else:
                        self.press_key(res)
                else:
                    self.press_key(None)

            # Trains the model.
            if self.parent.train:
                splitter = self.parent.layout().itemAt(0).widget() \
                    .layout().itemAt(0).widget()
                splitter.replaceWidget(1, self.parent.progress)

                # Sends the data to the model to train it.
                data = []
                targets = []
                for i, key in enumerate(self.parent.ele_tree):
                    eles = self.parent.ele_tree[key]
                    for ele in eles:
                        target = [0] * len(self.parent.ele_tree.keys())
                        data.append(ele)
                        target[i] = 1
                        targets.append(target)

                # Updates our model. (saves it too)
                self.parent.model = models.train_ffn(data, targets, hidden_size, \
                    len(self.parent.ele_tree), self.progress, max_epoch=1000)
                
                # Exits this if.
                self.parent.train = False
                splitter.replaceWidget(1, self.parent.label)
                self.parent.predict  = self.parent.model is not None


class HandTracking:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(tuple(locals().values())[1:])
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img):
        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks is not None:
            for hand_lms in self.results.multi_hand_landmarks:
                connections = self.mp_hands.HAND_CONNECTIONS
                self.mp_draw.draw_landmarks(img, hand_lms, connections)
        return img

    def get_pos(self, point):
        b = w/4

        newx = point[0] * (w + b*2) - b
        newy = point[1] * (h + b*2) - b
        return [newx, newy]

    def find_landmarks(self):
        lms = []
        if self.results.multi_hand_landmarks is not None:
            for hand_lm in self.results.multi_hand_landmarks:
                hand = []
                for lm in hand_lm.landmark:
                    hand.append((lm.x, lm.y))
                lms.append(hand)
        return lms


if __name__ == '__main__':
    # Starts everything up.
    app = QApplication(sys.argv)
    app.setWindowIcon(app.style().standardIcon(QStyle.SP_ComputerIcon))
    win = CalibrationUI()
    win.show()
    sys.exit(app.exec_())