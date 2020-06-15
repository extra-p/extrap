#import logging

import sys
from PyQt5.QtWidgets import QApplication
from gui.MainWidget import MainWidget

def main():
    
    #TODO: add logging to the gui application
    
    app = QApplication(sys.argv)
    
    window = MainWidget()
    window.show()
    
    app.exec_()



if __name__ == "__main__":
    main()
    
