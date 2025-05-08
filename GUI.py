from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QWidget, QMessageBox, QLabel
from qfluentwidgets import *
from utils import answersheet_score

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(800, 560)
        Form.setMinimumSize(QtCore.QSize(800, 560))
        Form.setMaximumSize(QtCore.QSize(800, 560))
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(430, 250, 151, 51))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.password = LineEdit(Form)
        self.password.setGeometry(QtCore.QRect(580, 260, 211, 41))
        #self.password.setDocumentTitle("")
        #self.password.setOverwriteMode(False)
        self.password.setObjectName("password")
        self.welcom = QtWidgets.QLabel(Form)
        self.welcom.setGeometry(QtCore.QRect(460, 60, 331, 101))
        self.welcom.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.welcom.setLineWidth(1)
        self.welcom.setMidLineWidth(0)
        self.welcom.setTextFormat(QtCore.Qt.AutoText)
        self.welcom.setScaledContents(False)
        self.welcom.setAlignment(QtCore.Qt.AlignCenter)
        self.welcom.setObjectName("welcom")
        self.id = LineEdit(Form)
        self.id.setGeometry(QtCore.QRect(580, 200, 211, 41))
        #self.id.setFrameShape(QtWidgets.QFrame.NoFrame)
        #self.id.setUndoRedoEnabled(True)
        self.id.setObjectName("id")
        self.background_figure = QtWidgets.QLabel(Form)
        self.background_figure.setGeometry(QtCore.QRect(0, 30, 431, 491))
        self.background_figure.setText("")
        self.background_figure.setPixmap(QtGui.QPixmap('images/background.PNG'))
        self.background_figure.setAlignment(QtCore.Qt.AlignCenter)
        self.background_figure.setObjectName("background_figure")
        self.log_on = PrimaryPushButton(Form)
        self.log_on.setGeometry(QtCore.QRect(480, 350, 311, 81))
        self.log_on.setObjectName("log_on")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(430, 190, 151, 51))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "登录界面-安鹏宇 庄翱 刘骏才"))
        self.label_3.setText(_translate("Form", "请输入密码"))
        self.password.setPlaceholderText(_translate("Form", "默认为六个6"))
        self.welcom.setText(_translate("Form", "欢迎使用答题卡计分系统！"))
        self.id.setPlaceholderText(_translate("Form", "默认为六个6"))
        self.log_on.setText(_translate("Form", "登录"))
        self.label_2.setText(_translate("Form", "请输入学号"))

class Ui_SecondForm(object):
    def setupUi(self, Form):
        Form.setObjectName("SecondForm")
        Form.resize(400, 300)
        self.label = QLabel(Form)
        self.label.setGeometry(100, 100, 200, 50)
        self.label.setText("欢迎来到答题卡评分系统")
        self.label.setObjectName("label")

class SecondForm(QWidget, Ui_SecondForm):
    def __init__(self):
        super(SecondForm, self).__init__()
        self.setupUi(self)
        self.startChecking.clicked.connect(self.on_startChecking_clicked)
        self.image_answers = ""  # 存储从图片中获取的答案序列
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1500, 1150)
        Form.setMinimumSize(QtCore.QSize(1500, 1150))
        Form.setMaximumSize(QtCore.QSize(1500, 1150))
        self.tip = QtWidgets.QLabel(Form)
        self.tip.setGeometry(QtCore.QRect(810, 30, 671, 201))
        self.tip.setAlignment(QtCore.Qt.AlignCenter)
        self.tip.setObjectName("tip")
        self.startChecking = PrimaryPushButton(Form)
        self.startChecking.setGeometry(QtCore.QRect(810, 730, 671, 161))
        self.startChecking.setObjectName("startChecking")
        self.uploadFig = CustomUploadButton(Form)
        self.uploadFig.setGeometry(QtCore.QRect(810, 290, 671, 161))
        self.uploadFig.setObjectName("uploadFig")
        self.if_upload = QtWidgets.QPlainTextEdit(Form)
        self.if_upload.setGeometry(QtCore.QRect(810, 510, 671, 161))
        self.if_upload.setReadOnly(True)
        self.if_upload.setObjectName("if_upload")
        self.results = QtWidgets.QPlainTextEdit(Form)
        self.results.setGeometry(QtCore.QRect(810, 950, 671, 161))
        self.results.setReadOnly(True)
        self.results.setObjectName("results")
        self.A71_80 = QtWidgets.QPlainTextEdit(Form)
        self.A71_80.setGeometry(QtCore.QRect(130, 860, 461, 65))
        self.A71_80.setObjectName("A71_80")
        self.L81_85 = QtWidgets.QLabel(Form)
        self.L81_85.setGeometry(QtCore.QRect(10, 980, 100, 51))
        self.L81_85.setMinimumSize(QtCore.QSize(100, 0))
        self.L81_85.setObjectName("L81_85")
        self.A81_85 = QtWidgets.QPlainTextEdit(Form)
        self.A81_85.setGeometry(QtCore.QRect(130, 970, 461, 65))
        self.A81_85.setObjectName("A81_85")
        self.A61_70 = QtWidgets.QPlainTextEdit(Form)
        self.A61_70.setGeometry(QtCore.QRect(130, 740, 461, 65))
        self.A61_70.setObjectName("A61_70")
        self.L71_80 = QtWidgets.QLabel(Form)
        self.L71_80.setGeometry(QtCore.QRect(10, 860, 100, 51))
        self.L71_80.setMinimumSize(QtCore.QSize(100, 0))
        self.L71_80.setObjectName("L71_80")
        self.A31_40 = QtWidgets.QPlainTextEdit(Form)
        self.A31_40.setGeometry(QtCore.QRect(130, 380, 461, 65))
        self.A31_40.setObjectName("A31_40")
        self.L41_50 = QtWidgets.QLabel(Form)
        self.L41_50.setGeometry(QtCore.QRect(10, 501, 100, 51))
        self.L41_50.setMinimumSize(QtCore.QSize(100, 0))
        self.L41_50.setObjectName("L41_50")
        self.A41_50 = QtWidgets.QPlainTextEdit(Form)
        self.A41_50.setGeometry(QtCore.QRect(130, 500, 461, 65))
        self.A41_50.setObjectName("A41_50")
        self.L51_60 = QtWidgets.QLabel(Form)
        self.L51_60.setGeometry(QtCore.QRect(10, 621, 100, 51))
        self.L51_60.setMinimumSize(QtCore.QSize(100, 0))
        self.L51_60.setObjectName("L51_60")
        self.A51_60 = QtWidgets.QPlainTextEdit(Form)
        self.A51_60.setGeometry(QtCore.QRect(130, 620, 461, 65))
        self.A51_60.setObjectName("A51_60")
        self.L61_70 = QtWidgets.QLabel(Form)
        self.L61_70.setGeometry(QtCore.QRect(10, 741, 100, 51))
        self.L61_70.setMinimumSize(QtCore.QSize(100, 0))
        self.L61_70.setObjectName("L61_70")
        self.A01_10 = QtWidgets.QPlainTextEdit(Form)
        self.A01_10.setGeometry(QtCore.QRect(130, 30, 461, 65))
        self.A01_10.setUndoRedoEnabled(False)
        self.A01_10.setReadOnly(False)
        self.A01_10.setOverwriteMode(False)
        self.A01_10.setMaximumBlockCount(0)
        self.A01_10.setObjectName("A01_10")
        self.L1_10 = QtWidgets.QLabel(Form)
        self.L1_10.setGeometry(QtCore.QRect(10, 31, 100, 61))
        self.L1_10.setMinimumSize(QtCore.QSize(100, 0))
        self.L1_10.setObjectName("L1_10")
        self.L11_20 = QtWidgets.QLabel(Form)
        self.L11_20.setGeometry(QtCore.QRect(10, 151, 100, 51))
        self.L11_20.setMinimumSize(QtCore.QSize(100, 0))
        self.L11_20.setObjectName("L11_20")
        self.A11_20 = QtWidgets.QPlainTextEdit(Form)
        self.A11_20.setGeometry(QtCore.QRect(130, 150, 461, 65))
        self.A11_20.setObjectName("A11_20")
        self.L21_30 = QtWidgets.QLabel(Form)
        self.L21_30.setGeometry(QtCore.QRect(10, 261, 100, 51))
        self.L21_30.setMinimumSize(QtCore.QSize(100, 0))
        self.L21_30.setObjectName("L21_30")
        self.A21_30 = QtWidgets.QPlainTextEdit(Form)
        self.A21_30.setGeometry(QtCore.QRect(130, 260, 461, 65))
        self.A21_30.setObjectName("A21_30")
        self.L31_40 = QtWidgets.QLabel(Form)
        self.L31_40.setGeometry(QtCore.QRect(10, 381, 100, 51))
        self.L31_40.setMinimumSize(QtCore.QSize(100, 0))
        self.L31_40.setObjectName("L31_40")
        self.score11_20 = QtWidgets.QPlainTextEdit(Form)
        self.score11_20.setGeometry(QtCore.QRect(610, 150, 101, 65))
        self.score11_20.setObjectName("score11_20")
        self.score01_10 = QtWidgets.QPlainTextEdit(Form)
        self.score01_10.setGeometry(QtCore.QRect(610, 30, 101, 65))
        self.score01_10.setObjectName("score01_10")
        self.score21_30 = QtWidgets.QPlainTextEdit(Form)
        self.score21_30.setGeometry(QtCore.QRect(610, 260, 101, 65))
        self.score21_30.setObjectName("score21_30")
        self.score31_40 = QtWidgets.QPlainTextEdit(Form)
        self.score31_40.setGeometry(QtCore.QRect(610, 380, 101, 65))
        self.score31_40.setObjectName("score31_40")
        self.score41_50 = QtWidgets.QPlainTextEdit(Form)
        self.score41_50.setGeometry(QtCore.QRect(610, 500, 101, 65))
        self.score41_50.setObjectName("score41_50")
        self.score51_60 = QtWidgets.QPlainTextEdit(Form)
        self.score51_60.setGeometry(QtCore.QRect(610, 620, 101, 65))
        self.score51_60.setObjectName("score51_60")
        self.score61_70 = QtWidgets.QPlainTextEdit(Form)
        self.score61_70.setGeometry(QtCore.QRect(610, 740, 101, 65))
        self.score61_70.setObjectName("score61_70")
        self.score71_80 = QtWidgets.QPlainTextEdit(Form)
        self.score71_80.setGeometry(QtCore.QRect(610, 860, 101, 65))
        self.score71_80.setObjectName("score71_80")
        self.score81_85 = QtWidgets.QPlainTextEdit(Form)
        self.score81_85.setGeometry(QtCore.QRect(610, 970, 101, 65))
        self.score81_85.setObjectName("score81_85")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "计分界面-by 安鹏宇 庄翱 刘骏才"))
        self.tip.setText(_translate("Form", "欢迎使用本答题卡计分系统 ! ! !\n\n请输入参考答案及每题占分"))
        self.startChecking.setText(_translate("Form", "开始计分"))
        self.uploadFig.setText(_translate("Form", "上传答题卡图片"))
        self.if_upload.setPlaceholderText(_translate("Form", "显示是否上传成功"))
        self.results.setPlaceholderText(_translate("Form", "显示计分结果"))
        self.A71_80.setPlaceholderText(_translate("Form", "10个选项，请以大写形式输入，不要有空格"))
        self.L81_85.setText(_translate("Form", "81-85"))
        self.A81_85.setPlaceholderText(_translate("Form", "5个选项"))
        self.A61_70.setPlaceholderText(_translate("Form", "10个选项，请以大写形式输入，不要有空格"))
        self.L71_80.setText(_translate("Form", "71-80"))
        self.A31_40.setPlaceholderText(_translate("Form", "10个选项，请以大写形式输入，不要有空格"))
        self.L41_50.setText(_translate("Form", "41-50"))
        self.A41_50.setPlaceholderText(_translate("Form", "10个选项，请以大写形式输入，不要有空格"))
        self.L51_60.setText(_translate("Form", "51-60"))
        self.A51_60.setPlaceholderText(_translate("Form", "10个选项，请以大写形式输入，不要有空格"))
        self.L61_70.setText(_translate("Form", "61-70"))
        self.A01_10.setPlaceholderText(_translate("Form", "10个选项,请以大写形式输入，不要有空格"))
        self.L1_10.setText(_translate("Form", "1-10"))
        self.L11_20.setText(_translate("Form", "11-20"))
        self.A11_20.setPlaceholderText(_translate("Form", "10个选项，请以大写形式输入，不要有空格"))
        self.L21_30.setText(_translate("Form", "21-30"))
        self.A21_30.setPlaceholderText(_translate("Form", "10个选项，请以大写形式输入，不要有空格"))
        self.L31_40.setText(_translate("Form", "31-40"))
        self.score11_20.setPlaceholderText(_translate("Form", "单题分值"))
        self.score01_10.setPlaceholderText(_translate("Form", "单题分值"))
        self.score21_30.setPlaceholderText(_translate("Form", "单题分值"))
        self.score31_40.setPlaceholderText(_translate("Form", "单题分值"))
        self.score41_50.setPlaceholderText(_translate("Form", "单题分值"))
        self.score51_60.setPlaceholderText(_translate("Form", "单题分值"))
        self.score61_70.setPlaceholderText(_translate("Form", "单题分值"))
        self.score71_80.setPlaceholderText(_translate("Form", "单题分值"))
        self.score81_85.setPlaceholderText(_translate("Form", "单题分值"))
        
    def process_image(self):
        # 调用answersheet_score函数处理图片并显示结果
        if hasattr(self.uploadFig, 'file_path'):
            self.image_answers = answersheet_score(self.uploadFig.file_path)
            
    def get_answers_and_scores(self):
        answers = ""
        scores = {}
        # 构建标准答案序列
        for i in range(0, 8): 
            # 构建每组答案的键名，并获取答案
            group_key = f"A{i}1_{(i+1)*10}"
            widget = getattr(self, group_key, None)
            if widget:
                answers += widget.toPlainText()
            else:
                self.results.setPlainText(f"请确保输入标准答案时没有遗漏")
            
            # 构建每组单题分值的键名，并获取分值
            score_key = f"score{i}1_{(i+1)*10}"
            score_widget = getattr(self, score_key, None)
            if score_widget:
                scores[f"{i}1-{(i+1)*10}"] = int(score_widget.toPlainText())
            else:
                self.results.setPlainText(f"请确保9组答案都输入了每题分值")

        # 处理最后一组（81到85的选项和分值）
        last_group_key = "A81_85"
        last_group_widget = getattr(self, last_group_key, None)
        if last_group_widget:
            answers += last_group_widget.toPlainText()
        else:
            self.results.setPlainText(f"请确保输入标准答案时没有遗漏")
        
        last_score_key = "score81_85"
        last_score_widget = getattr(self, last_score_key, None)
        if last_score_widget:
            scores["81-85"] = int(last_score_widget.toPlainText())
        else:
            self.results.setPlainText(f"请确保9组答案都输入了每题分值")
        return answers, scores
    
    def on_startChecking_clicked(self):
        standard_answers, scores = self.get_answers_and_scores()
        total_score = 0
        if len(self.image_answers) == 85 and len(standard_answers) == 85:
            for i in range(85):
                if self.image_answers[i] == standard_answers[i]:
                    # 根据题目编号确定分数的键名
                    if i < 80:  # 前80题，每10题一组
                        j=i//10
                        score_key = f"{j}1-{(j+1)*10}"
                    else:  # 最后5题
                        score_key = "81-85"
                    # 累加分数
                    if score_key in scores:
                        total_score += scores[score_key]
        if total_score==0:
            self.results.setPlainText(f"总分数：{total_score},请检查:\n1.最后一个答案输入框输入“5”个选项\n2.答案选项请以“大写”格式输入\n3.确保每组答案的每题分值已填写")
        else:
            self.results.setPlainText(f"总分数：{total_score}")  # 显示总分数

class LoginForm(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.log_on.clicked.connect(self.check_credentials)

    def check_credentials(self):
        # 检查用户名和密码是否都是6个6
        if self.id.text() == '666666' and self.password.text() == '666666':
            QMessageBox.information(self, '登录成功', '你已成功登录')
            self.second_form = SecondForm()  # 创建第二个界面的实例
            self.second_form.show()
            self.hide()  # 隐藏当前登录界面
        else:
            QMessageBox.warning(self, '登录失败', '用户名或密码错误')

class CustomUploadButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super(CustomUploadButton, self).__init__(parent)
        self.clicked.connect(self.upload_file)
        self.file_path = ""   # 存储成功上传的图片路径

    def upload_file(self):
        # 弹出文件选择框
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "", "Image files (*.jpg)", options=options)
        if fileName:
            self.check_and_display_upload_status(fileName)

    def check_and_display_upload_status(self, fileName):
        # 检查文件名是否符合要求
        if fileName.endswith('source/1.jpg') or fileName.endswith('source/2.jpg') or fileName.endswith('source/3.jpg'):
            self.parent().if_upload.setPlainText('成功上传')
            self.file_path = fileName  # 更新成功上传的图片路径
            self.parent().process_image()  # 处理图片，获取扫描到的答案
        else:
            self.parent().if_upload.setPlainText(f'上传失败,此时你上传的文件路径为{fileName},请上传本文件中相对路径为source/1.jpg、source/2.jpg或source/3.jpg的文件')