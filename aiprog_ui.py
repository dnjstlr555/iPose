from asyncio.windows_events import NULL
from calendar import c
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from numpy import char
from brain import *
from numpy.random import shuffle

dataset=Project(load=True, updateKmeans=True, k=9)

form_class = uic.loadUiType("aiprog_ui1.ui")[0]
form_subwindow = uic.loadUiType("aiprog_ui2.ui")[0]
form_Recomand = uic.loadUiType("aiprog_ui3.ui")[0]
form_Last4 = uic.loadUiType("aiprog_ui4.ui")[0]
form_Last8 = uic.loadUiType("aiprog_ui5.ui")[0]

class MainClass(QMainWindow, form_class):
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.show()

    def btn_main_to_second(self):
        self.hide()
        self.second = SubClass()               
        self.close()
    
    def four(self):
        self.second.send_number(4,0)
    
    def eight(self):
        self.second.send_number(8,0)

class SubClass(QDialog, QWidget, form_subwindow):
    def __init__(self):
        super(SubClass, self).__init__()
        self.setupUi(self)
        self.show()
        self.qImg = QPixmap()
        self.count = 0
        self.max_num = 0
        self.select_num = 0
        self.selected_images = []
        center=dataset.kMeansCenter()
        shuffle(center)
        self.img1 = center[0]
        self.img2 = center[1]
        self.img3 = center[2]
        self.img4 = center[3]
        self.img5 = center[4]
        self.img6 = center[5]
        self.img7 = center[6]
        self.img8 = center[7]
        self.img9 = center[8]

        self.images  = [self.img1,
        self.img2,
        self.img3,
        self.img4,
        self.img5,
        self.img6,
        self.img7,
        self.img8,
        self.img9]

        self.recomand_pose("pose")

    def recomand_pose(self, pose):
        recomand_pose = pose
        #서로다른 9개의 포즈를 추천하는 함수 !!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.qImg.load(self.img1)
        self.img_box_1.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(self.img2)
        self.img_box_2.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(self.img3)
        self.img_box_3.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(self.img4)
        self.img_box_4.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(self.img5)
        self.img_box_5.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(self.img6)
        self.img_box_6.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(self.img7)
        self.img_box_7.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(self.img8)
        self.img_box_8.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(self.img9)
        self.img_box_9.setPixmap(self.qImg.scaled(174,194))

    def send_number(self,maximum ,count_num):
        self.max_num = maximum
        self.count = count_num
        self.count_down()

    def send_selected_images1(self, images):
        self.selected_images = images
    
    def count_down(self):
        txt = str(self.count) + "/" + str(self.max_num)
        self.count_label.setText(txt)

    def pose_pop_up(self):
        buttonReply = QMessageBox.information(self, '알림', "이 포즈로 추천받겠습니까?", 
        QMessageBox.Yes | QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            self.recomand()

    def reset_pop_up(self):
        buttonReply = QMessageBox.information(self, '알림', "정말로 리셋하시겠습니까?", 
        QMessageBox.Yes | QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            self.reset()

    def p_one(self):
        self.select_num=0
        self.pose_pop_up()
    def p_two(self):
        self.select_num=1
        self.pose_pop_up()
    def p_three(self):
        self.select_num=2
        self.pose_pop_up()
    def p_four(self):
        self.select_num=3
        self.pose_pop_up()
    def p_five(self):
        self.select_num=4
        self.pose_pop_up()
    def p_six(self):
        self.select_num=5
        self.pose_pop_up()
    def p_seven(self):
        self.select_num=6
        self.pose_pop_up()
    def p_eight(self):
        self.select_num=7
        self.pose_pop_up()
    def p_nine(self):
        self.select_num=8
        self.pose_pop_up()


    def r_one(self):
        self.select_num=0
        self.reset_pop_up()
    def r_two(self):
        self.select_num=1
        self.reset_pop_up()
    def r_three(self):
        self.select_num=2
        self.reset_pop_up()
    def r_four(self):
        self.select_num=3
        self.reset_pop_up()
    def r_five(self):
        self.select_num=4
        self.reset_pop_up()
    def r_six(self):
        self.select_num=5
        self.reset_pop_up()
    def r_seven(self):
        self.select_num=6
        self.reset_pop_up()
    def r_eight(self):
        self.select_num=7
        self.reset_pop_up()
    def r_nine(self):
        self.select_num=8
        self.reset_pop_up()
    def r_ten(self):
        self.select_num=9
        self.reset_pop_up()
    
    def reset(self):
        #해당 이미지를 다른 이미지로 바꾸는 함수, 9는 모든 사진을 바꾼다는 의미!!!!!!!!!!!!!!!!!!!!!!!
        center=dataset.kMeansCenter()
        shuffle(center)
        self.img1 = center[0]
        self.img2 = center[1]
        self.img3 = center[2]
        self.img4 = center[3]
        self.img5 = center[4]
        self.img6 = center[5]
        self.img7 = center[6]
        self.img8 = center[7]
        self.img9 = center[8]

    def random(self):
        #number의 수 만큼 랜덤한 포즈를 추천!!!!!!!!!!!!!!!!
        number = self.max_num - self.count

        #랜덤한 포즈를 추천받고 끝낸다는 의미니 맨마지막 코드는 꼭 들어가야 함
        self.go_last_page()
    
    def recomand(self):           
        self.third = Recomand()
        self.third.send_number(self.max_num, self.count)
        self.third.recomand_imags(self.images[self.select_num])
        self.third.send_selected_images2(self.selected_images)
        self.third.count_down2()
        self.close()
        
    def stop(self):
        self.go_last_page()

    def go_last_page(self):
        if self.max_num == 4:
            self.hide()                
            self.last_4 = Last4()
            self.last_4.get_images(self.selected_images)
            self.close()
        else:
            self.hide()              
            self.last_8 = Last8()
            self.last_8.get_images(self.selected_images)
            self.close()

class Recomand(QDialog, QWidget, form_Recomand):
    def __init__(self):
        super(Recomand, self).__init__()
        self.setupUi(self)
        self.show()
        self.second = SubClass
        self.count = 0
        self.max_num = 0
        self.selected_images = []
        self.img1 = "./img/246-4.jpg"
        self.img2 = "./img/255-0.jpg"
        self.img3 = "./img/246-4.jpg"
        self.img4 = "./img/255-0.jpg"
        self.img5 = "./img/246-4.jpg"
        self.img6 = "./img/255-0.jpg"
        self.img7 = "./img/246-4.jpg"
        self.img8 = "./img/255-0.jpg"
        self.img9 = "./img/246-4.jpg"
        self.images  = [self.img1,
        self.img2,
        self.img3,
        self.img4,
        self.img5,
        self.img6,
        self.img7,
        self.img8,
        self.img9]

    def recomand_imags(self, image):
        recomand_image = image
        if image==None: return
        print(image)
        #image와 비슷한 포즈를 추천하는 함수 !!!!!!!!!!!!!!!!!!!!!!!!!!!
        simphoto=[i[0] for i in dataset.GetSimPhoto(image)]
        self.img1 = simphoto[0]
        self.img2 = simphoto[1]
        self.img3 = simphoto[2]
        self.img4 = simphoto[3]
        self.img5 = simphoto[4]
        self.img6 = simphoto[5]
        self.img7 = simphoto[6]
        self.img8 = simphoto[7]
        self.img9 = simphoto[8]
        self.images  = [self.img1,
        self.img2,
        self.img3,
        self.img4,
        self.img5,
        self.img6,
        self.img7,
        self.img8,
        self.img9]
        self.img_button_1.setStyleSheet('border-image:url(' + self.img1 + ');border:0px;')
        self.img_button_2.setStyleSheet('border-image:url(' + self.img2 + ');border:0px;')
        self.img_button_3.setStyleSheet('border-image:url(' + self.img3 + ');border:0px;')
        self.img_button_4.setStyleSheet('border-image:url(' + self.img4 + ');border:0px;')
        self.img_button_5.setStyleSheet('border-image:url(' + self.img5 + ');border:0px;')
        self.img_button_6.setStyleSheet('border-image:url(' + self.img6 + ');border:0px;')
        self.img_button_7.setStyleSheet('border-image:url(' + self.img7 + ');border:0px;')
        self.img_button_8.setStyleSheet('border-image:url(' + self.img8 + ');border:0px;')
        self.img_button_9.setStyleSheet('border-image:url(' + self.img9 + ');border:0px;')


    def send_number(self, maxaimum, count_num):
        self.max_num = maxaimum
        self.count = count_num

    def one(self):
        self.select_image(0)

    def two(self):
        self.select_image(1)
    
    def three(self):
        self.select_image(2)

    def four(self):
        self.select_image(3)

    def five(self):
        self.select_image(4)

    def six(self):
        self.select_image(5)

    def seven(self):
        self.select_image(6)

    def eight(self):
        self.select_image(7)

    def nine(self):
        self.select_image(8)

    def send_selected_images2(self, images):
        self.selected_images = images

    def select_image(self, num):
        self.selected_images += [self.images[num]]
        self.count += 1
        self.count_down2()

    def count_down2(self):
        txt = str(self.count) + "/" + str(self.max_num)
        self.count_label.setText(txt)
        if self.max_num == self.count:
            self.go_last_page()

    def btn_third_to_second(self):
        self.second = SubClass()
        self.second.send_number(self.max_num, self.count)
        self.second.send_selected_images1(self.selected_images)
        self.close()

    def go_last_page(self):
        print(self.selected_images,"HOLA")
        if self.max_num == 4:
            self.hide()                
            self.last_4 = Last4()
            self.last_4.get_images(self.selected_images)
            self.close()
        else:
            self.hide()              
            self.last_8 = Last8()
            self.last_8.get_images(self.selected_images)
            self.close()

class Last4(QDialog, QWidget, form_Last4):
    def __init__(self):
        super(Last4, self).__init__()
        self.setupUi(self)
        self.show()
        self.qImg = QPixmap()
    
    def get_images(self, images):
        print(images)
        if len(images) != 4:
            i = len(images)
            while i != 4:
                images += ["./img/0000-0000.jpg"]
                i += 1
        img1 = images[0]
        img2 = images[1]
        img3 = images[2]
        img4 = images[3]
        self.qImg.load(img1)
        self.img_box_1.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img2)
        self.img_box_2.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img3)
        self.img_box_3.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img4)
        self.img_box_4.setPixmap(self.qImg.scaled(174,194))

class Last8(QDialog, QWidget, form_Last8):
    def __init__(self):
        super(Last8, self).__init__()
        self.setupUi(self)
        self.show()
        self.qImg = QPixmap()
    
    def get_images(self, images):
        if len(images) != 8:
            i = len(images)
            while i != 8:
                images += ["./img/0000-0000.jpg"]
                i += 1
        img1 = images[0]
        img2 = images[1]
        img3 = images[2]
        img4 = images[3]
        img5 = images[4]
        img6 = images[5]
        img7 = images[6]
        img8 = images[7]
        self.qImg.load(img1)
        self.img_box_1.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img2)
        self.img_box_2.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img3)
        self.img_box_3.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img4)
        self.img_box_4.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img5)
        self.img_box_5.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img6)
        self.img_box_6.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img7)
        self.img_box_7.setPixmap(self.qImg.scaled(174,194))
        self.qImg.load(img8)
        self.img_box_8.setPixmap(self.qImg.scaled(174,194))

if __name__ == "__main__" :
    app = QApplication(sys.argv) 
    window = MainClass() 
    app.exec_()