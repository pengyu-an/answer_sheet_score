import sys
from PyQt5.QtWidgets import QApplication
from utils import *
from GUI import *


#打开UI界面
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { font-family: 'Arial'; font-size: 10pt; font-weight: bold}")
    login_form = LoginForm()
    login_form.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()