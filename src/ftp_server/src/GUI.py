import numpy as np
from numpy import random
from datetime import datetime
from datetime import timedelta

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QTextEdit, QSlider

import json
import os
import analysis

class MainWindow(QWidget):

    figure = plt.figure()

    def __init__(self):
        super().__init__()

        self.save_folder = ''
        self.startSession = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        self.template_file_name = 'conf.json'

        self.input_param = QLineEdit(self)
        self.input_param_button = QPushButton('Load template from json',self)
        self.data_table_label = QLabel('Path to data folder', self)
        self.data_table_field = QLineEdit(self)
        self.name_label = QLabel('Full Name', self)
        self.name_field = QLineEdit(self)
        self.class_label = QLabel('Classifier', self)
        self.class_field = QLineEdit(self)
        self.regr_label = QLabel('Regression', self)
        self.regr_field = QLineEdit(self)

        self.path_to_save_label = QLabel('Path to save Result', self)
        self.path_to_save_field = QLineEdit(self)

        self.predict_button = QPushButton('New prediction',self)
        self.view_progress_button = QPushButton('View Progress',self)
        self.preview_button = QPushButton('View Diagnosis',self)
        self.saveTemplate_button = QPushButton('Save template',self)

        self._dataPath = ''
        self._userName = 'NAME'
        self._classifier = 'Classifier'
        self._regression = 'Regression'
        self._pathToSave = ''
        self._preditPath = ''

        self.canvas = FigureCanvas(self.figure)
        self.plot_label = QLabel('Information plot', self)
        self.log_text_field = QTextEdit(self)
        self.log_prev_label = QLabel('Log Aidut',self)

        self.initUI()


    def initUI(self):

        self.setWindowTitle('MedHack')

        left_layout = QGridLayout()

        self.preview_button.clicked.connect(self.view_plt)
        self.view_progress_button.clicked.connect(self.view_plt)

        self.saveTemplate_button.clicked.connect(self.save_template)
        self.input_param_button.clicked.connect(self.load_data)

        self.predict_button.clicked.connect(self.prediction)

        left_layout.addWidget(self.input_param,0,0,1,1)
        left_layout.addWidget(self.input_param_button,0,1,1,1)
        left_layout.addWidget(self.data_table_label,1,0,1,1)
        left_layout.addWidget(self.data_table_field,1,1,1,1)
        left_layout.addWidget(self.name_label,2,0,1,1)
        left_layout.addWidget(self.name_field,2,1,1,1)
        left_layout.addWidget(self.class_label,3,0,1,1)
        left_layout.addWidget(self.class_field,3,1,1,1)
        left_layout.addWidget(self.regr_label,4,0,1,1)
        left_layout.addWidget(self.regr_field,4,1,1,1)

        left_layout.addWidget(self.path_to_save_label,5,0,1,1)
        left_layout.addWidget(self.path_to_save_field,5,1,1,1)

        left_layout.addWidget(self.preview_button,6,0,1,1)
        left_layout.addWidget(self.view_progress_button,6,1,1,1)

        left_layout.addWidget(self.predict_button,7,0,1,1)
        left_layout.addWidget(self.saveTemplate_button,7,1,1,1)

        right_layout = QGridLayout()

        right_layout.addWidget(self.plot_label, 0,0,1,2)
        right_layout.addWidget(self.canvas,1,0,1,2)

        main_layout = QHBoxLayout()

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        log_layout = QGridLayout()

        self.log_text_field.setReadOnly(True)

        log_layout.addWidget(self.log_prev_label,0,0,1,1)
        log_layout.addWidget(self.log_text_field,1,0,1,4)


        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(log_layout)

        self.setLayout(layout)
        self.show()


    def prediction(self):

        self.readAllData()

        name, time, dists, svm, marks = analysis.return_data('sample',[30,60,50])

        os.chdir('../../src/')

        self._userName = name
        self.name_field.setText(self._userName)

        result = 'Pathology find'

        if len(svm) > 0 :
            result += ': '
            for p in range(0,len(svm)):
                result += marks[p] + '-' + str(svm[p]) + ','
            result = result[:len(result)-1]

        else:
            result = 'Nothing'
            marks = [' ']
            svm = [0]

        self.viewEvent(self._userName + ' have a ' + result)

        self.save_data_base(name, marks, svm, time, dists)
        self.viewPlot('View Diagnosis')



    def load_data(self):

        try:
            if len(self.input_param.text()) > 0:

                conf = json.load(open(self.input_param.text(),'r'))

                self.data_table_field.setText(conf["Path"])
                self.name_field.setText(conf["User"])
                self.class_field.setText(conf["Classifier"])
                self.regr_field.setText(conf["Regression"])

        except Exception as e:
            self.viewEvent("Error: " + str(e))

    def save_template(self):

        if len(self.input_param.text()) > 0:
            self.template_file_name = self.input_param.text()

        else: self.template_file_name = 'conf.json'

        f = open(self.template_file_name, 'w')

        f.write('{\n  \"Path\": \"' + self.data_table_field.text() + '\",\n')
        f.write('  \"User\": \"' + self.name_field.text() + '\",\n')
        f.write('  \"Classifier\": \"' + self.class_field.text()+ '\",\n')
        f.write('  \"Regression\": \"' + self.regr_field.text()+ '\"\n')
        f.write('}')

        self.viewEvent('All data saved in ' + self.template_file_name + ' file')

        f.close()

    def load_data_base(self, name, _type = 1):

        f = []

        y = []
        x_label = []

        try:
            f = open(self._dataPath + 'data.csv', 'r')
        except:
            self.viewEvent('Missing data.csv file in' + self._dataPath + ' directy!')
            return y,x_label

        findFlag = False

        for line in f:
            line = line.rstrip('\n').split(';')

            if line[0] == name:

                data = line[_type].split(',')

                findFlag = True

                for k in data:
                    k = k.split('-')

                    if _type == 1:
                        y.append(float(k[1]))

                    if _type == 2:
                        y.append([ float(x) for x in k[1].split('_')])

                    x_label.append(k[0])

                break

        f.close()

        return y, x_label

    def save_data_base(self,name, marks, prob, time, dists):

        f = None

        fileFlag = False

        try:
            f = open(self._dataPath + 'data.csv', 'r')
            fileFlag = True
        except:
            pass


        findFlag = False

        lines = []

        if fileFlag:
            for line in f:
                line = line.rstrip('\n')

                _l = line.split(';')

                if _l[0] == name:

                    findFlag = True

                    _l[1] = ''

                    for i in range(0, len(marks)):
                        _l[1] += marks[i] + '-' + "%.4f" % prob[i] + ','

                    _l[1] = _l[1][:len(_l[1]) -1]

                    if len(_l[2]) > 2:
                        _l[2] += ','

                    _l[2] += time.strftime("%d:%m:%Y:%H:%M") + '-' + "%.4f" % dists[0] + '_' + "%.4f" % dists[1] + '_' + "%.4f" % dists[2]

                lines.append(_l[0] + ';' + _l[1] + ';' + _l[2] + '\n')

            f.close()

        if not findFlag:
            line = name + ';'

            for i in range(0, len(marks)):
                line += marks[i] + '-' + "%.4f" % prob[i] + ','

            line = line[:len(line)-1]

            line += ';' + time.strftime("%d:%m:%Y:%H:%M") + '-' +  "%.4f" % dists[0] + '_' + "%.4f" % dists[1] + '_' + "%.4f" % dists[2] + '\n'

            lines.append(line)

        try:
            f = open(self._dataPath + 'data.csv', 'w')
        except:
            self.viewEvent('Missing data.csv file in' + self._dataPath + ' directy!')
            return False

        for line in lines:
            f.write(line)

        f.close()

    def view_plt(self):

        source = self.sender()
        self.viewPlot(source.text())


    def viewPlot(self, action):

        self.readAllData()

        self.figure.clear()

        ax = self.figure.add_subplot(111)

        if action == 'View Progress':

            data,x_label = self.load_data_base(self._userName, _type = 2)

            if len(data) == 0:
                self.viewEvent('This person (' + self._userName + ') don\'t have a info in data base!')
                return False

            x = [ i for i in range(0, len(x_label))]

            x_1 = [ i[0] for i in data]
            y_1 = [ i[1] for i in data]
            z_1 = [ i[2] for i in data]

            norm_x = [30 for i in data]
            norm_y = [60 for i in data]
            norm_z = [50 for i in data]

            ax.plot(x, x_1, 'b', label = 'X- макс. откл. 30')
            ax.plot(x ,y_1 ,'y', label= 'Y- макс. откл. 60')
            ax.plot(x, z_1, 'g', label = 'Z - макс. откл. 50')
            #ax.plot(x, norm_x, 'b--', label = 'Norm X')
            #ax.plot(x, norm_y, 'y--', label = 'Norm Y')
            #ax.plot(x,norm_z, 'g--', label = 'Norm Z')
            ax.legend()
            ax.set_xticks(x)
            #ax.set_xticklabels(x_label, rotation=90, fontsize = 10, va='bottom', ha='right')
            ax.set_xticklabels([' ' for i in range(0,len(x))])
            ax.set_yticks([0,30,60, 50])

            ax.set_xlabel('Time')
            ax.set_ylabel('Distanse for normal steps')
            #ax.set_xticklabels(x_label)
            ax.set_title('Reabilitation progress')

            self.canvas.draw()

            self.viewEvent("Preview plot was build")

        elif action == 'View Diagnosis':

            y,x_label = self.load_data_base(self._userName, _type = 1)

            if len(y) == 0:
                self.viewEvent('This person (' + self._userName + ') don\'t have a info in data base!')
                return False

            x = [ i for i in range(0, len(x_label))]

            ax.bar(x,y)

            ax.set_xticks(x)
            ax.set_xticklabels(x_label, rotation=90, fontsize = 10, va='bottom', ha='right')
            ax.set_yticks([0, 1])

            ax.set_xlabel('Diagnosis')
            ax.set_ylabel('Probobility')
            ax.set_title('Diagnosis recomendation')

            self.canvas.draw()

            self.viewEvent("Preview plot was build")


    def readAllData(self):
        try:

            self._dataPath =  self.data_table_field.text()
            self._userName = self.name_field.text()
            self._classifier = self.class_field.text()
            self._regression = self.regr_field.text()
            self._pathToSave = self.path_to_save_field.text()

            line = self._pathToSave.split('/')

            self.save_folder = ''
            if len(line) > 1:
                for x in range(0, len(line) - 1):
                    self.save_folder += line[x] + '/'

        except Exception as e:
           self.viewEvent("Error: " + str(e))

    def viewEvent(self,msg, date = True, save = True):

        date_str = ""

        if date:
            date_str = "[" + datetime.now().strftime("%d:%m:%Y %H:%M:%S") + "]: "

        msg = self.log_text_field.toPlainText() + date_str + msg + "\n"
        self.log_text_field.setText(msg)

        if save:
            try:
                log = open(self.save_folder + self.startSession + '_log.txt', 'w')
                log.write(self.log_text_field.toPlainText())
                log.close()
            except Exception as e:
                viewEvent(str(e), save = False)



def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

    return

if __name__ == '__main__':
    main()
