import tkinter.messagebox as messagebox
from tkinter import *
import threading
import tkinter
import random
import time
import math
import PIL
import LOS


class simulation:

    def __init__(self, master=None):
        self.root = master
        self.root.geometry('1000x800')
        self.root.title('SIMULATION')
        self.para = {}
        self.para['MT_x'],self.para['MT_y'],self.para['MT']=500, 350,[500,350]#init value for convenience
        self.para['BT_Los_x'], self.para['BT_Los_y'] = [300, 700, 300,700], [250, 250, 450,450]
        self.para['BT_Nlos_x'], self.para['BT_Nlos_y'] = [600,500,400], [200, 150, 500]
        self.para['BT_x'],self.para['BT_y']=self.para['BT_Los_x']+self.para['BT_Nlos_x'],self.para['BT_Los_y']+self.para['BT_Nlos_y']
        self.para['ave'],self.para['sigma'],self.para['Number_Los']=0,2,4
        self.para['lamda'],self.para['Number_Nlos']=2,3
        self.para['Number']=7
        self.para['map']='map.png'
        self.Home()

    def get_BT(self, e1, e2,e3,e4) :
        BT_x, BT_y, Dis, MPE, dis_ = [], [], [], [], []
        try:
            for i in range(len(e1)):
                x = float(e1[i].get())
                y = float(e2[i].get())
                BT_x.append(x)
                BT_y.append(y)
                d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
                Dis.append(d)
                n = math.log(d) * random.gauss(self.para['ave'], self.para['sigma'])
                MPE.append(n)
                dis_.append(d + n)

            self.para['BT_Los_x'] = BT_x
            self.para['BT_Los_y'] = BT_y
            self.para['Dis_Los'] = Dis
            self.para['MPE_Los'] = MPE
            self.para['dis_Los'] = dis_

            BT_x, BT_y, Dis, MPE, dis_ = [], [], [], [], []
            for i in range(len(e3)):
                x = float(e3[i].get())
                y = float(e4[i].get())
                BT_x.append(x)
                BT_y.append(y)
                d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
                Dis.append(d)
                n = math.log(d) * self.para['lamda']*random.expovariate(self.para['lamda'])
                MPE.append(n)
                dis_.append(d + n)

            self.para['BT_Nlos_x'] = BT_x
            self.para['BT_Nlos_y'] = BT_y
            self.para['Dis_Nlos'] = Dis
            self.para['MPE_Nlos'] = MPE
            self.para['dis_Nlos'] = dis_
            self.para['dis_']=self.para['dis_Nlos']+self.para['dis_Los']
            self.para['BT_x'], self.para['BT_y'] = self.para['BT_Los_x'] + self.para['BT_Nlos_x'], self.para['BT_Los_y'] + self.para['BT_Nlos_y']
            random.shuffle(self.para['BT_x'])
            random.shuffle(self.para['BT_y'])
            #print(self.para)
            self.Home()
            #self.PlotStation()
        except:
            messagebox.showwarning("warning!")

    def Gauss(self,e1,e2):
        try:
            self.para['ave'] = float(e1.get())
            self.para['sigma'] = float(e2.get())
            #print(self.para)
        except:
            messagebox.showwarning("warning!")

    def expoen(self,e1):
        try:
            self.para['lamda'] = float(e1.get())
            #print(self.para)
        except:
            messagebox.showwarning("warning!")

    def MT_pos(self,e1,e2):
        try:
            self.para['MT_x'] = float(e1.get())
            self.para['MT_y'] = float(e2.get())
        except:
            messagebox.showwarning("warning!")

    def GetNum_Los(self,e1):
        try:
            self.para['Number_Los'] = int(e1.get())
            self.para['Number']=self.para['Number_Los']+self.para['Number_Nlos']
        except:
            messagebox.showwarning("warning!")

    def GetNum_Nlos(self,e1):
        try:
            self.para['Number_Nlos'] = int(e1.get())
            self.para['Number'] = self.para['Number_Los'] + self.para['Number_Nlos']
        except:
            messagebox.showwarning("warning!")

    def drawbase(self):
        self.frame1.img_bg = PIL.ImageTk.PhotoImage(PIL.Image.open(self.para['map']).resize((1000, 680)))
        mt_png = self.frame1.create_image((0, 0), anchor='nw')
        self.frame1.itemconfig(mt_png, image=self.frame1.img_bg)

        self.frame1.img_mt = PIL.ImageTk.PhotoImage(PIL.Image.open('estimate.png').resize((30, 30)))
        mt_png = self.frame1.create_image((self.para['MT_x'], self.para['MT_y']), anchor='nw')
        self.frame1.itemconfig(mt_png, image=self.frame1.img_mt)  # 内存保护机制，自动销毁图片内存，需要将图片绑定到canavas本身

        self.frame1.img_bt_los = PIL.ImageTk.PhotoImage(PIL.Image.open('BT.png').resize((30, 30)))
        for x, y in zip(self.para['BT_Los_x'], self.para['BT_Los_y']):
            bt_png = self.frame1.create_image((x, y), anchor='nw')
            self.frame1.itemconfig(bt_png, image=self.frame1.img_bt_los)

        self.frame1.img_bt_Nlos = PIL.ImageTk.PhotoImage(PIL.Image.open('BT_N.png').resize((30, 30)))
        for x, y in zip(self.para['BT_Nlos_x'], self.para['BT_Nlos_y']):
            bt_png = self.frame1.create_image((x, y), anchor='nw')
            self.frame1.itemconfig(bt_png, image=self.frame1.img_bt_Nlos)

    def Circle(self):
        self.frame1.delete(tkinter.ALL)
        BT_Los_x, BT_Los_y, Dis_Los, MPE_Los, dis_Los = [], [], [], [], []
        BT_Nlos_x, BT_Nlos_y, Dis_Nlos, MPE_Nlos, dis_Nlos = [], [], [], [], []
        for i in range(self.para['Number']):
            x = self.para['MT_x'] + math.sin(2 * math.pi / self.para['Number'] * i) * min(self.para['MT_x'],1000 - self.para['MT_y']) * 0.75
            y = self.para['MT_y'] + math.cos(2 * math.pi / self.para['Number'] * i) * min(self.para['MT_y'],680 - self.para['MT_y']) * 0.75
            d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
            tag=random.randint(0,1)
            if tag==0:
                BT_Los_x.append(x)
                BT_Los_y.append(y)
                Dis_Los.append(d)
                n = math.log(d) * random.gauss(self.para['ave'], self.para['sigma'])
                MPE_Los.append(n)
                dis_Los.append(d + n)
            else:
                BT_Nlos_x.append(x)
                BT_Nlos_y.append(y)
                Dis_Nlos.append(d)
                n = math.log(d) * self.para['lamda']*random.expovariate(self.para['lamda'])
                MPE_Nlos.append(n)
                dis_Nlos.append(d + n)

        self.para['BT_Los_x'],self.para['BT_Nlos_x']  = BT_Los_x,BT_Nlos_x
        self.para['BT_Los_y'],self.para['BT_Nlos_y'] = BT_Los_y,BT_Nlos_y
        self.para['Dis_Los'],self.para['Dis_Nlos'] = Dis_Los,Dis_Nlos
        self.para['MPE_Los'],self.para['MPE_Nlos'] = MPE_Los,MPE_Nlos
        self.para['dis_Los'],self.para['dis_Nlos'] = dis_Los,dis_Nlos
        self.para['BT_x'], self.para['BT_y'] = self.para['BT_Los_x'] + self.para['BT_Nlos_x'], self.para['BT_Los_y'] + self.para['BT_Nlos_y']
        random.shuffle(self.para['BT_x'])
        random.shuffle(self.para['BT_y'])
        self.drawbase()

    def Randd(self):
        self.frame1.delete(tkinter.ALL)
        BT_Los_x, BT_Los_y, Dis_Los, MPE_Los, dis_Los = [], [], [], [], []
        BT_Nlos_x, BT_Nlos_y, Dis_Nlos, MPE_Nlos, dis_Nlos = [], [], [], [], []
        for i in range(self.para['Number']):
            x = random.random() * 1000
            y = random.random() * 680
            d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
            tag = random.randint(0, 1)
            if tag==0:
                BT_Los_x.append(x)
                BT_Los_y.append(y)
                Dis_Los.append(d)
                n = math.log(d) * random.gauss(self.para['ave'], self.para['sigma'])
                MPE_Los.append(n)
                dis_Los.append(d + n)
            else:
                BT_Nlos_x.append(x)
                BT_Nlos_y.append(y)
                Dis_Nlos.append(d)
                n = math.log(d) * self.para['lamda']*random.expovariate(self.para['lamda'])
                MPE_Nlos.append(n)
                dis_Nlos.append(d + n)

        self.para['BT_Los_x'], self.para['BT_Nlos_x'] = BT_Los_x, BT_Nlos_x
        self.para['BT_Los_y'], self.para['BT_Nlos_y'] = BT_Los_y, BT_Nlos_y
        self.para['Dis_Los'], self.para['Dis_Nlos'] = Dis_Los, Dis_Nlos
        self.para['MPE_Los'], self.para['MPE_Nlos'] = MPE_Los, MPE_Nlos
        self.para['dis_Los'], self.para['dis_Nlos'] = dis_Los, dis_Nlos
        self.para['BT_x'], self.para['BT_y'] = self.para['BT_Los_x'] + self.para['BT_Nlos_x'], self.para['BT_Los_y'] + self.para['BT_Nlos_y']
        random.shuffle(self.para['BT_x'])
        random.shuffle(self.para['BT_y'])
        self.drawbase()

    def Set_XJTU(self):
        self.para['map']='map1.png'
        self.Home()

    def Set_Cartoon(self):
        self.para['map']='map.png'
        self.Home()

    def Set_World(self):
        self.para['map']='world.png'
        self.Home()

    def Home(self):
        menu = Menu(self.root)
        self.root.config(menu=menu)

        scenario = Menu(menu)
        menu.add_cascade(label='Scenario', menu=scenario)
        scenario.add_command(label='Home', command=self.Home)

        draw=Menu(menu)
        menu.add_cascade(label='Draw', menu=draw)
        draw.add_command(label='Circle', command=self.Circle)
        draw.add_command(label='Random', command=self.Randd)
        draw.add_command(label='Input', command=self.input)

        method=Menu(menu)
        LOS=Menu(method)
        NLOS = Menu(method)

        ml=Menu(LOS)
        ml.add_command(label='AML',command=self.AML)
        ml.add_command(label='Newton_Gauss', command=self.newton_gaussian)

        ls = Menu(LOS)
        ls.add_command(label='LLS_1',command=self.LLS_1)
        ls.add_command(label='LLS_AVE', command=self.LLS_AVER)
        ls.add_command(label='LLS_RS', command=self.LLS_RS)

        sdr = Menu(LOS)
        sdr.add_command(label='R_LS', command=self.R_LS)

        menu.add_cascade(label='Method', menu=method)
        method.add_cascade(label='LOS',menu=LOS)
        method.add_cascade(label='NLOS',menu=NLOS)

        LOS.add_cascade(label='ML', menu=ml)
        LOS.add_cascade(label='LS', menu=ls)
        LOS.add_cascade(label='SDR', menu=sdr)

        para=Menu(menu)
        menu.add_cascade(label='Para', menu=para)
        para.add_command(label='set_para',command=self.Set_Para)

        map = Menu(menu)
        menu.add_cascade(label='Map', menu=map)
        map.add_command(label='XJTU', command=self.Set_XJTU)
        map.add_command(label='Cartoon', command=self.Set_Cartoon)
        map.add_command(label='World', command=self.Set_World)


        self.frame1 = Canvas(self.root)

        self.frame1.config(bg='white', height=680, width=1000)
        Label(self.frame1, text='frame1').place(in_=self.frame1, anchor=NW)
        self.frame1.place(x=0, y=0)
        self.frame1.img_bg = PIL.ImageTk.PhotoImage(PIL.Image.open(self.para['map']).resize((1000, 680)))
        mt_png = self.frame1.create_image((0, 0), anchor='nw')
        self.frame1.itemconfig(mt_png, image=self.frame1.img_bg)


    def Set_Para(self):
        self.para_frame = Frame(self.root)
        self.para_frame.config(bg='#e8c974', height=280, width=300)
        Label(self.para_frame, text='para').place(in_=self.para_frame, anchor=NW)
        self.para_frame.place(x=400, y=200)
        Number_Los = Entry(self.para_frame)
        Number_Los.place(in_=self.para_frame, anchor=NW, x=60, y=40, width=85)
        Button(self.para_frame, text='Lnum', command=lambda: self.GetNum_Los(Number_Los)).place(in_=self.para_frame, anchor=NW, x=160,y=40,width=50)
        Number_Nlos = Entry(self.para_frame)
        Number_Nlos.place(in_=self.para_frame, anchor=NW, x=60, y=80, width=85)
        Button(self.para_frame, text='NLnum', command=lambda: self.GetNum_Nlos(Number_Nlos)).place(in_=self.para_frame,anchor=NW, x=160, y=80,width=50)

        Ave = Entry(self.para_frame)
        Ave.place(in_=self.para_frame, anchor=NW, x=60, y=120, width=40)
        Sigma = Entry(self.para_frame)
        Sigma.place(in_=self.para_frame, anchor=NW, x=105, y=120, width=40)
        Button(self.para_frame, text='Gauss', command=lambda: self.Gauss(Ave, Sigma)).place(in_=self.para_frame, anchor=NW,x=160,y=120, width=50)

        Lamda = Entry(self.para_frame)
        Lamda.place(in_=self.para_frame, anchor=NW, x=60, y=160, width=85)
        Button(self.para_frame, text='Exp', command=lambda: self.expoen(Lamda)).place(in_=self.para_frame,anchor=NW, x=160, y=160,width=50)

        MT_x = Entry(self.para_frame)
        MT_x.place(in_=self.para_frame, anchor=NW, x=60, y=200, width=40)
        MT_y = Entry(self.para_frame)
        MT_y.place(in_=self.para_frame, anchor=NW, x=105, y=200, width=40)
        Button(self.para_frame, text='MT pos', command=lambda: self.MT_pos(MT_x, MT_y)).place(in_=self.para_frame, anchor=NW,x=160, y=200, width=50)
        Button(self.para_frame, bg='#f5ce42', text='Home', command=self.Home).place(in_=self.para_frame, anchor=NW, x=120, y=240, width=80)

    def input(self):
        number_Los = int(self.para['Number_Los'])
        number_Nlos = int(self.para['Number_Nlos'])

        self.frame2_1 = Frame(self.root)
        self.frame2_1.config(bg='white', height=800, width=1000)
        Label(self.frame2_1, text='frame1').place(in_=self.frame2_1, anchor=NW)
        self.frame2_1.place(x=0, y=0)
        E1_x, E1_y = [], []
        E2_x, E2_y = [], []
        skip = int(400 / number_Los)

        Label(self.frame2_1, text='LOS').place(in_=self.frame2_1, anchor=NW,x=250,y=100)
        for i in range(number_Los):
            e_t1 = Entry(self.frame2_1, bg='#f5ce42')
            e_t1.place(in_=self.frame2_1, anchor=NW, x=250, y=160 + i * skip, width=50)
            e_t2 = Entry(self.frame2_1, bg='#f5ce42')
            e_t2.place(in_=self.frame2_1, anchor=NW, x=350, y=160 + i * skip, width=50)
            E1_x.append(e_t1)
            E1_y.append(e_t2)

        Label(self.frame2_1, text='NLOS').place(in_=self.frame2_1, anchor=NW, x=550, y=100)
        for i in range(number_Nlos):
            e_t1 = Entry(self.frame2_1, bg='#f5ce42')
            e_t1.place(in_=self.frame2_1, anchor=NW, x=550, y=160 + i * skip, width=50)
            e_t2 = Entry(self.frame2_1, bg='#f5ce42')
            e_t2.place(in_=self.frame2_1, anchor=NW, x=650, y=160 + i * skip, width=50)
            E1_x.append(e_t1)
            E1_y.append(e_t2)

        Button(self.frame2_1, bg='#f5ce42', text='Confirm', command=lambda: self.get_BT(E1_x, E1_y,E2_x,E2_y)).place(
            in_=self.frame2_1, anchor=NW, x=800, y=400, width=100)
        Button(self.frame2_1, bg='#f5ce42', text='Home', command=self.Home).place(
            in_=self.frame2_1, anchor=NW, x=800, y=300, width=100)

    def draw_espos(self,es_pos):
        self.frame1.img_es_aml = PIL.ImageTk.PhotoImage(PIL.Image.open('MT.png').resize((30, 30)))
        mt_png = self.frame1.create_image((es_pos[0], es_pos[1]), anchor='nw')
        self.frame1.itemconfig(mt_png, image=self.frame1.img_es_aml)
        self.frame1.create_text((es_pos[0] + 10, es_pos[1] - 10), anchor='nw',
                                text='[' + str(es_pos[0]) + ',' + str(es_pos[1]) + ']', fill='#f20242')

    def AML(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        aml = LOS.Program(self.para['MT'], self.para['BT_Los_x'], self.para['BT_Los_y'], self.para['Dis_Los'], self.para['dis_Los'])
        es_pos = aml.estimate()
        self.draw_espos(es_pos)

    def LLS_1(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        lls1=LOS.Program(self.para['MT'], self.para['BT_Los_x'], self.para['BT_Los_y'], self.para['Dis_Los'], self.para['dis_Los'])
        es_pos = lls1.LLS_1_E()
        self.draw_espos(es_pos)

    def LLS_AVER(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        lls_aver=LOS.Program(self.para['MT'], self.para['BT_Los_x'], self.para['BT_Los_y'], self.para['Dis_Los'], self.para['dis_Los'])
        es_pos = lls_aver.LLS_AVE_E()
        self.draw_espos(es_pos)

    def LLS_RS(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        lls_rs=LOS.Program(self.para['MT'], self.para['BT_Los_x'], self.para['BT_Los_y'], self.para['Dis_Los'], self.para['dis_Los'])
        es_pos = lls_rs.LLS_RS_E()
        self.draw_espos(es_pos)

    def R_LS(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        s_ls=LOS.Program(self.para['MT'], self.para['BT_Los_x'], self.para['BT_Los_y'], self.para['Dis_Los'], self.para['dis_Los'])
        es_pos = s_ls.R_LS_E()
        self.draw_espos(es_pos)

    def NewTon_Gauss(self):
        N_G=LOS.Program(self.para['MT'], self.para['BT_Los_x'], self.para['BT_Los_y'], self.para['Dis_Los'], self.para['dis_Los'])
        es_pos = N_G.NewTon_Gauss()
        length=len(es_pos)
        self.frame1.img_es_gn = PIL.ImageTk.PhotoImage(PIL.Image.open('MT.png').resize((30, 30)))
        for i in range(length-1):
            mt_png = self.frame1.create_image((es_pos[i][0], es_pos[i][1]), anchor='nw')
            self.frame1.itemconfig(mt_png, image=self.frame1.img_es_gn)
            time.sleep(1)
            self.frame1.delete(mt_png)
        mt_png = self.frame1.create_image((es_pos[length-1][0], es_pos[length-1][1]), anchor='nw')
        self.frame1.itemconfig(mt_png, image=self.frame1.img_es_gn)
        self.frame1.create_text((es_pos[length-1][0] + 10, es_pos[length-1][1] - 10), anchor='nw',
                                text='[' + str(es_pos[length-1][0]) + ',' + str(es_pos[length-1][1]) + ']', fill='#f20242')

    def newton_gaussian(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        T=threading.Thread(target=self.NewTon_Gauss)
        T.start()
#---------------------------------------------------------------------------NLOS
def main():
    root = Tk()
    s=simulation(root)
    mainloop()


if __name__ == "__main__":
    main()
