import sys
import cv2
import time
import mediapipe as mp
from pynput import keyboard
import math as m
import json
import os

from PyQt5.QtCore import QThread, Qt, QObject, QEvent, QTimer
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage

# Custom imports.
import models

# TODO: Let user see what is set edit and remove them.
# TODO: Make calibration window not use popup windows for new elements.
# TODO: Add newtwork to manage classification.
# TODO: Let window be resizeable.

class CalibrationUI(QMainWindow):
    '''
    Brings together the menu and the calibration images.
    '''
    def __init__(self):
        super().__init__()

        # Checks if the file exists.
        if not os.path.exists('data.json'):
            with open("data.json", "w+") as f:
                f.write('{}')
                self.key_tree = json.load(f)
        else:
            # Updates key tree if there is data.
            with open("data.json", "r") as f:
                self.key_tree = json.load(f)

        # Loads the model.
        self.options = list(self.key_tree.keys())
        self.model = models.FFN(80, len(self.options))
        self.predict = len(self.options) > 0
        self.train = False

        # Window settings.
        self.setWindowTitle("Hand Mouse Calibration")
        self.resize(1280, 720)
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray_quit = False

        # Image target.
        self.label = QLabel(self)

        # Keyboard controler.
        self.key_ctrl = keyboard.Controller()
        self.get_key = False
        self.get_macro = False

        # Thread for content to update on.
        self.thread = QThread()
        self.content = Content(self)
        self.content.moveToThread(self.thread)
        self.thread.started.connect(self.content.update_loop)
        
        # Setup window.
        self.menu = Menu(self)
        self.setMenuBar(self.menu)
        self.setCentralWidget(self.label)
        self.thread.start()

        # Tray setup
        tray_menu = QMenu(self)
        cali_action = QAction('Calibrate', self)
        cali_action.triggered.connect(self.cali_action)
        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(self.tray_quit_action)
        tray_menu.addAction(cali_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)
        self.tray.setContextMenu(tray_menu)

    def closeEvent(self, event):
        if self.tray_quit:
            self.content.stop()
            self.thread.quit()
        else:
            event.ignore()
            self.tray.show()
            self.hide()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() and Qt.WindowMinimized:
                self.tray.show()
                self.hide()

    def cali_action(self):
        '''
        Brings the calibration window back.
        '''
        self.show()
        self.tray.hide()

    def tray_quit_action(self):
        '''
        Closes the window when the user quits it from the icon.
        '''
        self.tray_quit = True
        self.close()


class Menu(QMenuBar):
    '''
    Top menu to save settings.
    '''
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.set_key = SetKeyMenu(parent)
        self.set_macro = SetMacroMenu(parent)
        self.set_menu = None

        # Top of menu.
        file_menu = QMenu("File", self)
        set_menu = QMenu("Set", self)
        set_mouse = QMenu("Mouse", self)
        set_button = QAction("Key",self)
        set_button.triggered.connect(lambda: self.show_menu(self.set_key))
        set_macro = QAction("Macro", self)
        set_macro.triggered.connect(lambda: self.show_menu(self.set_macro))

        # Buttons to update mouse.
        set_tracking = QAction("Tracking", self)
        set_tracking.triggered.connect(lambda: self.update_action("MMOVE"))
        set_left_mouse = QAction("Left Button", self)
        set_left_mouse.triggered.connect(lambda: self.update_action("MLEFT"))
        set_right_mouse = QAction("Right Button", self)
        set_right_mouse.triggered.connect(lambda: self.update_action("MRIGHT"))
        set_scroll = QAction("Scroll", self)
        set_scroll.triggered.connect(lambda: self.update_action("MSCROLL"))

        # Save setup
        save_file = QAction("Save", self)
        save_file.triggered.connect(self.save_action)

        # Adds all options to the menu.
        set_mouse.addAction(set_tracking)
        set_mouse.addAction(set_left_mouse)
        set_mouse.addAction(set_right_mouse)
        set_mouse.addAction(set_scroll)
        set_menu.addMenu(set_mouse)
        set_menu.addAction(set_button)
        set_menu.addAction(set_macro)
        file_menu.addMenu(set_menu)
        file_menu.addAction(save_file)
        self.addMenu(file_menu)

    def save_action(self):
        '''
        Saves all elements to file.
        '''
        # Saves data.
        with open("data.json", "w+") as f:
            to_save = json.dumps(self.parent.key_tree, indent=4, sort_keys=True)
            f.write(to_save)

        # Updates tree based on the save.
        with open("data.json", "r") as f:
            self.parent.key_tree = json.load(f)

        # Updates options and title.
        self.parent.setWindowTitle("Hand Mouse Calibration")
        self.parent.options = list(self.parent.key_tree.keys())
        self.parent.train = True

    def update_action(self, action):
        '''
        Updates new key to the key tree.

        Parameters
        ----------
        action: string
            The key or button to save.

        Returns
        -------
        boolean: If it updated or not.
        '''
        self.parent.setWindowTitle("Hand Mouse Calibration*")
        self.pop_up = PopUp('Count Down', 'Place hand on screen in 3')
        QTimer.singleShot(1000, lambda: self.pop_up.set_text('Place hand on screen in 2'))
        QTimer.singleShot(2000, lambda: self.pop_up.set_text('Place hand on screen in 1'))
        QTimer.singleShot(3000, lambda: self.pop_up.set_text('Keep hand on screen and moving.'))
        QTimer.singleShot(3000, lambda: self.collect_data(action, 0))

    def collect_data(self, action, i, future_action=[]):
        '''
        Collects the data from the hand.
        '''
        if i != 10:
            # Gets hand and updates it.
            dist = self.parent.content.get_dists()
            if len(dist) > 0:
                temp = future_action.copy()
                temp.append(dist)
                QTimer.singleShot(500, lambda: self.collect_data(action, i+1, temp))
            # Cancels tree update.
            else:
                self.pop_up.set_text('Hand left screen, calibration canceled.')
                self.parent.menu.set_menu.updated.setText(action + " failed to update...")
                QTimer.singleShot(1000, lambda: self.pop_up.close())
        else:
            # Updates tree.
            self.parent.key_tree[action] = future_action
            self.parent.menu.set_menu.updated.setText(action + " updated...")
            self.pop_up.close()

    def show_menu(self, content):
        '''
        Shows the content as a pop up window
        '''
        self.parent.predict = False
        self.set_menu = content
        content.show()


class Content(QObject):
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
        self.listener = keyboard.Listener(on_press=self.on_press)
        
        # Camera settings
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
        self.listener.start()

    def on_press(self, key):
        res = str(key).strip("\'").replace("Key.", "").upper()
        
        # Prevents missing '.
        if res == "":
            res = "\'"

        if self.parent.get_key:
            self.parent.menu.set_key.text_box.setText(res)
        if self.parent.get_macro:
            self.parent.menu.set_macro.active.setText(res)

    def stop(self):
        '''
        Stops the video feed.
        '''
        self.running = False
        self.cam.release()
        self.listener.stop()
        cv2.destroyAllWindows()

    def get_dists(self):
        '''
        Function to get the current distances to save as a profile.
        
        Returns
        -------
        float[]: List of all distances between important joints.
        '''
        lms = self.ht.find_landmarks()
        dists = []

        for i in range(len(lms)):
            if len(lms[i]) >= 20:
                # Collects distances from joints.
                base = m.dist(lms[i][0], lms[i][5])
                dists.append(m.dist(lms[i][0], lms[i][1])/base)
                dists.append(m.dist(lms[i][1], lms[i][2])/base)
                dists.append(m.dist(lms[i][2], lms[i][3])/base)
                dists.append(m.dist(lms[i][3], lms[i][4])/base)
                dists.append(m.dist(lms[i][0], lms[i][5])/base)
                dists.append(m.dist(lms[i][5], lms[i][6])/base)
                dists.append(m.dist(lms[i][6], lms[i][7])/base)
                dists.append(m.dist(lms[i][7], lms[i][8])/base)
                dists.append(m.dist(lms[i][0], lms[i][9])/base)
                dists.append(m.dist(lms[i][9], lms[i][10])/base)
                dists.append(m.dist(lms[i][10], lms[i][11])/base)
                dists.append(m.dist(lms[i][11], lms[i][12])/base)
                dists.append(m.dist(lms[i][0], lms[i][13])/base)
                dists.append(m.dist(lms[i][13], lms[i][14])/base)
                dists.append(m.dist(lms[i][14], lms[i][15])/base)
                dists.append(m.dist(lms[i][15], lms[i][16])/base)
                dists.append(m.dist(lms[i][0], lms[i][17])/base)
                dists.append(m.dist(lms[i][17], lms[i][18])/base)
                dists.append(m.dist(lms[i][18], lms[i][19])/base)
                dists.append(m.dist(lms[i][19], lms[i][20])/base)

        return dists

    def update_loop(self):
        '''
        Gets the current image and displays it in the content.
        '''
        while self.running:
            _, frame = self.cam.read()
            frame = cv2.flip(frame, 1)

            # Checking if we are able to detect the hand...
            img = self.ht.find_hands(frame)

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
            
            # Make the hand prediction.
            if self.parent.predict and len(self.parent.options) > 0:
                dist = self.get_dists()
                if len(dist) >= 20:
                    dist.extend([0] * 20) if len(dist) < 40 else dist
                    pred = models.predict_ffn(dist, self.parent.model)
                    if pred[1] > 0.99:
                        print(self.parent.options[pred[0]])

            # Trains the model.
            if self.parent.train:
                self.parent.train = False
                if not self.parent.isHidden():
                    self.parent.label.setText('Training')

                # Sends the data to the model to train it.
                data = []
                targets = []
                for i, key in enumerate(self.parent.key_tree):
                    eles = self.parent.key_tree[key]
                    for ele in eles:
                        target = [0] * len(self.parent.options)
                        ele.extend([0] * 20) if len(ele) < 40 else ele
                        data.append(ele)
                        target[i] = 1
                        targets.append(target)

                # Updates our model. (saves it too)
                self.parent.model = models.train_ffn(data, targets, 80, len(self.parent.key_tree))


class SetKeyMenu(QWidget):
    def __init__(self, parent):
        '''
        Creates the set key menu.

        Parameters
        ----------
        parent: PyQtElement
            The very top level element of the menu system.
        '''
        super().__init__()
        self.setWindowTitle("Set Key")
        self.parent = parent
        self.parent.get_key = True

        # Set key button.
        set_key = QPushButton("Set Key", self)
        set_key.clicked.connect(lambda: self.parent.menu.update_action(self.text_box.text()))

        # Key display
        self.text_box = QLineEdit(self)
        self.text_box.setReadOnly(True)

        # Update display
        self.updated = QLabel("Waiting...",self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Press key to set:", self))
        layout.addWidget(self.text_box)
        layout.addWidget(self.updated)
        layout.addWidget(set_key)
        self.setLayout(layout)
        

    def closeEvent(self, event):
        '''
        Stops getting key strokes.

        Parameters
        ----------
        event: PyQtEvent
            The event that caused the event to fire.
        '''
        self.parent.get_key = False
        self.parent.predict = True


class SetMacroMenu(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.setWindowTitle('Set Macro')
        self.parent = parent
        self.parent.get_macro = True
        self.text_boxes = []
        self.keys = []
        self.delays = []
        self.target = 0
        self.active = None

        # Buttons.
        buttons = QWidget(self)
        but_lay = QHBoxLayout()
        set_macro = QPushButton('Set Macro', self)
        set_macro.clicked.connect(self.set_macro)
        add_key = QPushButton('Add Key',self)
        add_key.clicked.connect(self.add_key)
        del_key = QPushButton('Delet Key', self)
        del_key.clicked.connect(self.del_key)

        # Button layout.
        but_lay.addWidget(add_key)
        but_lay.addWidget(set_macro)
        but_lay.addWidget(del_key)
        buttons.setLayout(but_lay)
        

        # Update display
        self.updated = QLabel('Waiting...',self)
        self.label = QLabel('Editing Key 1:', self)

        # Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.updated)
        self.layout.addWidget(buttons)
        self.setLayout(self.layout)
        self.add_key()

    def add_key(self):
        '''
        Adds a key to the macro.
        '''
        # Key display
        key_in = QWidget(self)
        key_in_layout = QHBoxLayout()
        text_box = QLineEdit(key_in)
        text_box.setReadOnly(True)
        delay = QSpinBox(key_in)
        delay.setRange(0, 99999)
        ms = QLabel('ms',key_in)
        cur = len(self.text_boxes)
        edit = QPushButton(key_in)
        edit.setIcon(QIcon(QPixmap('assets/edit.png')))
        edit.clicked.connect(lambda: self.edit_btn(cur))

        # Sets the layout up.
        key_in_layout.addWidget(text_box,stretch=3)
        key_in_layout.addWidget(delay,stretch=2)
        key_in_layout.addWidget(ms,stretch=1)
        key_in_layout.addWidget(edit,stretch=1)
        key_in.setLayout(key_in_layout)

        # Updates key array and active.
        self.keys.append(key_in)
        self.delays.append(delay)
        self.text_boxes.append(text_box)
        self.active = text_box
        self.target = cur
        self.layout.insertWidget(len(self.text_boxes),key_in)
        self.label.setText(f'Editing Key {self.target+1}:')

    def set_macro(self):
        '''
        Sets the macro in the key tree.
        '''
        macro_str = 'Macro:'
        for i in range(len(self.keys)):
            macro_str += f' {self.text_boxes[i].text()}:{self.delays[i].value()}'
        self.parent.menu.update_action(macro_str)

    def del_key(self):
        '''
        Deletes a key.
        '''
        if len(self.keys) > 1:
            self.delays.pop(self.target)
            self.text_boxes.pop(self.target)
            self.keys.pop(self.target).deleteLater()
            
            if self.target >= len(self.keys):
                self.target = len(self.keys) - 1
            
            self.active = self.text_boxes[self.target]
            self.label.setText(f'Editing Key {self.target+1}:')

    def edit_btn(self, target):
        '''
        Informs the user of what they are
        currently editing and changes the target.

        Parameters
        ----------
        target: int
            The index of the text box that will be edited.
        '''
        self.active = self.text_boxes[target]
        self.label.setText(f'Editing Key {target+1}:')
        self.target = target

    def closeEvent(self, event):
        '''
        Stops getting key strokes.

        Parameters
        ----------
        event: PyQtEvent
            The event that caused the event to fire.
        '''
        self.parent.get_macro = False
        self.parent.predict = True


class PopUp(QWidget):
    '''
    Class display a popup menu.
    '''
    def __init__(self, title, text):
        '''
        Creates the initial popup.

        Parameters
        ----------
        title: str
            The title of the popup window.
        text: str
            The text to display in the popup window.
        '''
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle(title)
        layout = QHBoxLayout(self)
        self.label = QLabel(text, self)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.show()


    def set_text(self, text):
        self.label.setText(text)


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

    def find_landmarks(self):
        lms = []
        if self.results.multi_hand_landmarks is not None:
            for hand_lm in self.results.multi_hand_landmarks:
                hand = []
                for lm in hand_lm.landmark:
                    hand.append((lm.x, lm.y, lm.z))
                lms.append(hand)
        return lms

# Starts everything up.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(app.style().standardIcon(QStyle.SP_ComputerIcon))
    win = CalibrationUI()
    win.show()
    sys.exit(app.exec_())