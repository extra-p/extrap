#import logging

import sys
from PyQt5.QtWidgets import QApplication
from gui.MainWidget import MainWidget
from fileio.text_file_reader import read_text_file


def main():

    # TODO: add logging to the gui application

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    # app.setStyleSheet('QWidget{background:#333;color:#eee}')

    window = MainWidget()
    window.show()

    if len(sys.argv) == 3 and '--text' == sys.argv[1]:
        experiment = read_text_file(sys.argv[2])
        # call the modeler and create a function model
        window.model_experiment(experiment)

    app.exec_()


if __name__ == "__main__":
    main()
