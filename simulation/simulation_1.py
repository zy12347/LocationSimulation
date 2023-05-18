import tkinter.messagebox as messagebox
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import *
import numpy as np
import threading
import analysis
import tkinter
import random
import time
import math
import NLOS
import PIL
import LOS

class simulation:

    def __init__(self, master=None):
        self.root = master
        self.root.geometry('1000x800')
        self.root.title('SIMULATION')
        self.para = {}
        self.para['MT_x'], self.para['MT_y'] = 500, 350 #init value for convenience
        self.para['BT_Los_x'], self.para['BT_Los_y'] = [100, 800, 100, 800], [100, 100, 600, 600]
        self.para['BT_Nlos_x'], self.para['BT_Nlos_y'] = [600, 500, 400], [200, 150, 500]
        self.para['ave'], self.para['sigma'], self.para['Number_Los'] = 0, 5, 4
        self.para['lamda'], self.para['Number_Nlos'] = 5,3
        self.para['Number']=7
        self.para['t0']=random.random()*50
        self.para['map'] = 'White.png'
        self.para['BT_NLOS'], self.para['BT_LOS'], self.para['BT'] = [], [], []
        self.Home()

    def get_BT(self, e1, e2, e3, e4):
        BT_x, BT_y, BT_NLOS,BT_LOS = [], [], [],[]
        try:
            for i in range(len(e1)):
                x = float(e1[i].get())
                y = float(e2[i].get())
                BT_x.append(x)
                BT_y.append(y)
                d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
                n = random.gauss(self.para['ave'], self.para['sigma'])
                BT_LOS.append([x, y, d + n])

            self.para['BT_LOS']=BT_LOS
            self.para['BT_Los_x'] = BT_x
            self.para['BT_Los_y'] = BT_y

            BT_x, BT_y = [], []
            for i in range(len(e3)):
                x = float(e3[i].get())
                y = float(e4[i].get())
                BT_x.append(x)
                BT_y.append(y)
                d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
                n = (self.para['lamda']*np.random.exponential(self.para['lamda'])+random.gauss(self.para['ave'], self.para['sigma']))

            self.para['BT_Nlos_x'] = BT_x
            self.para['BT_Nlos_y'] = BT_y
            self.para['BT_NLOS']=BT_NLOS
            self.para['BT']=self.para['BT_LOS']+self.para['BT_NLOS']
            self.Home()
            self.drawbase()
        except:
            messagebox.showwarning("warning!",'input can not be none!')

    def Gauss(self,e1,e2):
        try:
            self.para['ave'] = float(e1.get())
            self.para['sigma'] = float(e2.get())
            #print(self.para)
        except:
            messagebox.showwarning("warning!",'input can not be none!')

    def expoen(self,e1):
        try:
            self.para['lamda'] = float(e1.get())
            #print(self.para)
        except:
            messagebox.showwarning("warning!",'input can not be none!')

    def MT_pos(self,e1,e2):
        try:
            self.para['MT_x'] = float(e1.get())
            self.para['MT_y'] = float(e2.get())
        except:
            messagebox.showwarning("warning!",'input can not be none!')

    def GetNum_Los(self,e1):
        try:
            if int(e1.get()) < 3:
                messagebox.showwarning('Warning!', "larger than 3!")
            else:
                self.para['Number_Los'] = int(e1.get())
                self.para['Number']=self.para['Number_Los']+self.para['Number_Nlos']
        except:
            messagebox.showwarning("warning!",'input can not be none!')

    def GetNum_Nlos(self,e1):
        try:
            self.para['Number_Nlos'] = int(e1.get())
            self.para['Number'] = self.para['Number_Los'] + self.para['Number_Nlos']
        except:
            messagebox.showwarning("warning!",'input can not be none!')

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

    def draw_espos(self,es_pos):
        self.frame1.img_es_aml = PIL.ImageTk.PhotoImage(PIL.Image.open('MT.png').resize((30, 30)))
        mt_png = self.frame1.create_image((es_pos[0], es_pos[1]), anchor='nw')
        self.frame1.itemconfig(mt_png, image=self.frame1.img_es_aml)
        self.frame1.create_text((es_pos[0] + 10, es_pos[1] - 10), anchor='nw',
                                text='[' + str(round(es_pos[0],1)) + ',' + str(round(es_pos[1],1)) + ']', fill='#f20242')

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

    def Circle(self):
        self.frame1.delete(tkinter.ALL)
        BT_NLOS,BT_LOS=[],[]
        BT_Los_x, BT_Los_y = [], []
        BT_Nlos_x, BT_Nlos_y = [], []
        rand_list = [x for x in range(self.para['Number'])]
        random.shuffle(rand_list)
        '''
        for i in rand_list[0:-3]:
            x = self.para['MT_x'] + math.sin(2 * math.pi / self.para['Number'] * i) * min(self.para['MT_x'],980 - self.para['MT_x']) * 0.75
            y = self.para['MT_y'] + math.cos(2 * math.pi / self.para['Number'] * i) * min(self.para['MT_y'],650 - self.para['MT_y']) * 0.75
            d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
            tag=random.randint(0,2)
            if tag==0:
                BT_Los_x.append(x)
                BT_Los_y.append(y)
                n = random.gauss(self.para['ave'], self.para['sigma'])
                BT_LOS.append([x, y, d + n])
            else:
                BT_Nlos_x.append(x)
                BT_Nlos_y.append(y)
                n = math.log(d) * (self.para['lamda']*np.random.exponential(self.para['lamda'])+random.gauss(self.para['ave'], self.para['sigma']))
                BT_NLOS.append([x, y, d + n])
        '''
        for i in rand_list:
            x = 400 + math.sin(2 * math.pi / self.para['Number'] * i) * 290
            y = 300 + math.cos(2 * math.pi / self.para['Number'] * i) * 290
            d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
            BT_Los_x.append(x)
            BT_Los_y.append(y)
            n = random.gauss(self.para['ave'], self.para['sigma'])
            BT_LOS.append([x, y, d + n])

        self.para['BT_Los_x'],self.para['BT_Nlos_x']  = BT_Los_x,BT_Nlos_x
        self.para['BT_Los_y'],self.para['BT_Nlos_y'] = BT_Los_y,BT_Nlos_y
        self.para['BT_NLOS']=BT_NLOS
        self.para['Number_Los']=len(BT_Los_x)
        self.para['Number_Nlos'] = len(BT_Nlos_x)
        self.para['BT_LOS']=BT_LOS
        self.para['BT'] = self.para['BT_LOS'] + self.para['BT_NLOS']
        self.drawbase()

    def Randd(self):
        self.frame1.delete(tkinter.ALL)
        BT_NLOS,BT_LOS=[],[]
        BT_Los_x, BT_Los_y= [], []
        BT_Nlos_x, BT_Nlos_y= [], []
        for i in range(self.para['Number']-3):
            x = random.random() * 900+50
            y = random.random() * 600+50
            d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
            tag = random.randint(0, 2)
            if tag==0:
                BT_Los_x.append(x)
                BT_Los_y.append(y)
                n = random.gauss(self.para['ave'], self.para['sigma'])
                BT_LOS.append([x,y,d+n])
            else:
                BT_Nlos_x.append(x)
                BT_Nlos_y.append(y)
                n = (self.para['lamda']*np.random.exponential(self.para['lamda'])+random.gauss(self.para['ave'], self.para['sigma']))
                BT_NLOS.append([x,y,d+n])

        for i in range(3):
            x = random.random() * 900 + 50
            y = random.random() * 600 + 50
            d = ((self.para['MT_x'] - x) ** 2 + (self.para['MT_y'] - y) ** 2) ** 0.5
            BT_Los_x.append(x)
            BT_Los_y.append(y)
            n = random.gauss(self.para['ave'], self.para['sigma'])
            BT_LOS.append([x, y, d + n])

        self.para['BT_LOS']=BT_LOS
        self.para['BT_Los_x'], self.para['BT_Nlos_x'] = BT_Los_x, BT_Nlos_x
        self.para['BT_Los_y'], self.para['BT_Nlos_y'] = BT_Los_y, BT_Nlos_y
        self.para['BT_NLOS']=BT_NLOS
        self.para['Number_Los'] = len(BT_Los_x)
        self.para['Number_Nlos'] = len(BT_Nlos_x)
        self.para['BT'] = self.para['BT_LOS'] + self.para['BT_NLOS']
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

    def Set_White(self):
        self.para['map']='White.png'
        self.Home()

    def Home(self):
        menu = Menu(self.root)
        self.root.config(menu=menu)

        scenario = Menu(menu)
        menu.add_cascade(label='Scenario', menu=scenario)
        scenario.add_command(label='Home', command=self.Home)

        Dynamic = Menu(menu)
        menu.add_cascade(label='Dynamic', menu=Dynamic)
        Dynamic.add_command(label='Dynamic', command=self.Dynamic)

        draw=Menu(menu)
        menu.add_cascade(label='Draw', menu=draw)
        draw.add_command(label='Circle', command=self.Circle)
        draw.add_command(label='Random', command=self.Randd)
        draw.add_command(label='Input', command=self.input)

        method=Menu(menu)
        NLOS = Menu(method)
        NLOS.add_command(label='RWGH', command=self.RWGH)
        NLOS.add_command(label='LLS', command=self.LLS_NLOS)

        LOS = Menu(method)
        ml=Menu(LOS)
        ml.add_command(label='AML',command=self.AML)
        ml.add_command(label='Newton_Gauss', command=self.newton_gaussian)

        ls = Menu(LOS)
        ls.add_command(label='LLS_1',command=self.LLS_1)
        ls.add_command(label='LLS_AVE', command=self.LLS_AVER)
        ls.add_command(label='LLS_RS', command=self.LLS_RS)

        sdr = Menu(LOS)
        sdr.add_command(label='R_LS', command=self.R_LS)
        sdr.add_command(label='SR_LS', command=self.SR_LS)

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
        map.add_command(label='White', command=self.Set_White)

        compare=Menu(menu)
        menu.add_cascade(label='Compare', menu=compare)
        compare.add_command(label='COMPARE',command=self.Compare)
        compare.add_command(label='HeatMap', command=self.Heatmap)
        compare.add_command(label='plot3D', command=self.plot3D)

        self.frame1 = Canvas(self.root)
        self.frame1.config(bg='white', height=680, width=1000)
        Label(self.frame1, text='frame1').place(in_=self.frame1, anchor=NW)
        self.frame1.place(x=0, y=0)
        self.frame1.img_bg = PIL.ImageTk.PhotoImage(PIL.Image.open(self.para['map']).resize((1000, 680)))
        mt_png = self.frame1.create_image((0, 0), anchor='nw')
        self.frame1.itemconfig(mt_png, image=self.frame1.img_bg)

    def DrawCar(self):
        interval, t_acc = 1, 0
        pos,v = [400,300],[0,0]

        BT_LOS1 = []
        for x_, y_ in zip(self.para['BT_Los_x'], self.para['BT_Los_y']):
            d = ((pos[0] - x_) ** 2 + (pos[1] - y_) ** 2) ** 0.5
            n = random.gauss(self.para['ave'], self.para['sigma'])
            BT_LOS1.append([x_, y_, d + n])
        self.para['BT_LOS'] = BT_LOS1
        lls1 = LOS.Program(self.para['BT_LOS'])
        es_pos_init = lls1.LLS_1_E()

        #state estimate paras
        F=np.array([[1,0,interval,0],[0,1,0,interval],[0,0,1,0],[0,0,0,1]])#transfer matrix
        q=0.5#energy power
        mean=[0,0,0,0]#noise mean
        Q=q*np.array([[interval**3/3,0,interval**2/2,0],[0,interval**3/3,0,interval**2/2],[interval**2/2,0,interval,0],[0,interval**2/2,0,interval]])#covariance
        init_state = [es_pos_init[0], es_pos_init[1], 0, 0]
        P_init = np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])

        while True:
            t_acc=t_acc+interval
            v=[100*math.cos(t_acc),100*math.sin(t_acc)]

            x=pos[0]+random.randint(-1,1)*v[0]*interval
            y=pos[1]+random.randint(-1,1)*v[1]*interval
            if x<10 or x>1000:
                v[0]=-v[0]
                x=pos[0]+v[0]*interval
            if y<10 or y>600:
                v[1]=-v[1]
                y=pos[1]+v[1]*interval
            self.Frame_Dynamic.img_mt = PIL.ImageTk.PhotoImage(PIL.Image.open('MT.png').resize((30, 30)))
            mt_png = self.Frame_Dynamic.create_image((x, y), anchor='nw')
            self.Frame_Dynamic.itemconfig(mt_png, image=self.Frame_Dynamic.img_mt)
            time.sleep(interval)
            self.Frame_Dynamic.create_line(pos[0],pos[1],x,y,fill='blue',width=2)
            pos=[x,y]
            print("pos:",pos)

            BT_LOS = []
            for x_, y_ in zip(self.para['BT_Los_x'], self.para['BT_Los_y']):
                d = ((pos[0] - x_) ** 2 + (pos[1] - y_) ** 2) ** 0.5
                n = random.gauss(self.para['ave'], self.para['sigma'])
                BT_LOS.append([x_, y_, d + n])
            self.para['BT_LOS'] = BT_LOS

            length = len(self.para['BT_LOS'])-1
            H, Z = [], []
            R = np.zeros((length,length))
            x0=self.para['BT_LOS'][0][0]
            y0=self.para['BT_LOS'][0][1]
            dis0=self.para['BT_LOS'][0][2]
            for i,bt in enumerate(self.para['BT_LOS'][1:]):
                H.append([-2 * (bt[0] - x0), -2 * (bt[1] - y0), 0, 0])
                Z.append(bt[2] ** 2 - dis0 ** 2 - (
                            (bt[0] ** 2 + bt[1] ** 2) - (bt[0] ** 2 + bt[1] ** 2)))
                R[i][i] = 4*(bt[2]-dis0)**2*self.para['sigma']
            H = np.array(H)
            Z = np.array(Z)

            state_noise = np.random.multivariate_normal(mean, Q)
            estimate_state = F.dot(init_state)
            print("estimate_pos:",estimate_state)
            P_estimate = F.dot(P_init).dot(F.T)+Q
            K = P_estimate.dot(H.T).dot(np.linalg.inv(H.dot(P_estimate).dot(H.T)+R))
            X_state = estimate_state+ K.dot(Z-H.dot(estimate_state))
            P_state = P_estimate - K.dot(H).dot(P_estimate)

            self.Frame_Dynamic.img_es = PIL.ImageTk.PhotoImage(PIL.Image.open('estimate.png').resize((30, 30)))
            es_png = self.Frame_Dynamic.create_image((X_state[0], X_state[1]), anchor='nw')
            self.Frame_Dynamic.itemconfig(es_png, image=self.Frame_Dynamic.img_es)
            self.Frame_Dynamic.create_line(init_state[0], init_state[1], X_state[0], X_state[1], fill='green', width=2)
            init_state = X_state
            P_init = P_state
            print("X_state:",X_state)

    def Dynamic(self):
        self.Frame_Dynamic = Canvas(self.root)
        self.Frame_Dynamic.config(bg='white', height=680, width=1000)
        Label(self.Frame_Dynamic, text='Dynamic').place(in_=self.Frame_Dynamic, anchor=NW)
        self.Frame_Dynamic.place(x=0, y=0)
        self.Frame_Dynamic.img_bg = PIL.ImageTk.PhotoImage(PIL.Image.open('map1.png').resize((1000, 680)))
        bg_png = self.Frame_Dynamic.create_image((0, 0), anchor='nw')
        self.Frame_Dynamic.itemconfig(bg_png, image=self.Frame_Dynamic.img_bg)

        self.Frame_Dynamic.img_bt_los = PIL.ImageTk.PhotoImage(PIL.Image.open('BT.png').resize((30, 30)))
        for x, y in zip(self.para['BT_Los_x'], self.para['BT_Los_y']):
            bt_png = self.Frame_Dynamic.create_image((x, y), anchor='nw')
            self.Frame_Dynamic.itemconfig(bt_png, image=self.Frame_Dynamic.img_bt_los)

        T = threading.Thread(target=self.DrawCar)
        T.start()

    def AML(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        aml = LOS.Program(self.para['BT_LOS'])
        es_pos = aml.estimate()
        self.draw_espos(es_pos)
        return es_pos

    def LLS_1(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        lls1=LOS.Program(self.para['BT_LOS'])
        es_pos = lls1.LLS_1_E()
        self.draw_espos(es_pos)
        return es_pos

    def LLS_AVER(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        lls_aver=LOS.Program(self.para['BT_LOS'])
        es_pos = lls_aver.LLS_AVE_E()
        self.draw_espos(es_pos)
        return es_pos

    def LLS_RS(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        lls_rs=LOS.Program(self.para['BT_LOS'])
        es_pos = lls_rs.LLS_RS_E()
        self.draw_espos(es_pos)
        return es_pos

    def R_LS(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        s_ls=LOS.Program(self.para['BT_LOS'])
        es_pos = s_ls.R_LS_E()
        self.draw_espos(es_pos)
        return es_pos

    def SR_LS(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        sr_ls=LOS.Program(self.para['BT_LOS'])
        es_pos = sr_ls.SR_LS()
        self.draw_espos(es_pos)
        return es_pos

    def NewTon_Gauss(self):
        N_G=LOS.Program(self.para['BT_LOS'])
        es_pos = N_G.NewTon_Gauss()
        length=len(es_pos)
        self.frame1.img_es_gn = PIL.ImageTk.PhotoImage(PIL.Image.open('MT.png').resize((30, 30)))
        for i in range(length-1):
            mt_png = self.frame1.create_image((es_pos[i][0], es_pos[i][1]), anchor='nw')
            self.frame1.itemconfig(mt_png, image=self.frame1.img_es_gn)
            time.sleep(1)
            self.frame1.delete(mt_png)
        self.draw_espos([es_pos[length-1][0], es_pos[length-1][1]])
        return [es_pos[length-1][0], es_pos[length-1][1]]

    def newton_gaussian(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        T=threading.Thread(target=self.NewTon_Gauss)
        T.start()

    def lsSteps(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        bt = self.para['BT_LOS']
        for i in range(len(bt)):
            bt[i][2] = bt[i][2] + self.para['t0']
        lsteps = LOS.Program_t(bt)
        es_pos = lsteps.LS_steps()
        self.draw_espos(es_pos)
        return es_pos

#---------------------------------------------------------------------------NLOS

    def LLS_NLOS(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        lls_nlos = NLOS.Program(self.para['BT'])
        es_pos = lls_nlos.LLS_NLOS()
        self.draw_espos(es_pos)
        return es_pos

    def RWGH(self):
        self.frame1.delete(tkinter.ALL)
        self.drawbase()
        rwgh=NLOS.Program(self.para['BT'])
        NLOS_pos,es_pos = rwgh.RSWG()
        for pos in NLOS_pos:
            self.frame1.create_text((pos[0] + 10, pos[1] - 10), anchor='nw',text='LOS',fill='#f20242')
        self.draw_espos(es_pos)
        return es_pos

#----------------------------------------------------------------------------------------------------Compare
    def Scenario_para(self):
        self.frame3_2 = Frame(self.frame3_1)
        self.frame3_2.config(bg='#bfd4f5', height=700, width=300)
        self.frame3_2.place(in_=self.frame3_1, x=20, y=50)
        Label(self.frame3_2, text='场景参数').place(in_=self.frame3_2, anchor=NW, x=0, y=0)

        columns = ("base", "x pos", "y pos")
        headers = ("基站", "X坐标", "Y坐标")
        info = [('MT', round(self.para['MT_x'], 1), round(self.para['MT_y'], 1)), ('LOS', '', '')]
        for i in range(self.para['Number_Los']):
            info.append(('BT' + str(i), round(self.para['BT_Los_x'][i], 1), round(self.para['BT_Los_y'][i], 1)))
        info.append(('NLOS', '', ''))

        for i in range(self.para['Number_Nlos']):
            info.append(('BT' + str(self.para['Number_Los'] + i), round(self.para['BT_Nlos_x'][i], 1),
                         round(self.para['BT_Nlos_y'][i], 1)))

        tv = ttk.Treeview(self.frame3_2, show="headings", columns=columns)
        for (column, header) in zip(columns, headers):
            tv.column(column, width=80, anchor="nw")
            tv.heading(column, text=header, anchor="nw")

        for i, data in enumerate(info):
            tv.insert('', i, values=data)

        tv.place(in_=self.frame3_2, anchor=NW, x=25, y=60)

        head=('误差','μ/λ','sigma')
        col=('noise','ave','sigma')
        tv1 = ttk.Treeview(self.frame3_2, show="headings", columns=col)
        for (column, header) in zip(col, head):
            tv1.column(column, width=80, anchor="nw")
            tv1.heading(column, text=header, anchor="nw")
        tv1.insert('',1,values=('Gauss',self.para['ave'],self.para['sigma']))
        tv1.insert('', 2, values=('Expoen', self.para['lamda'], ''))
        tv1.place(in_=self.frame3_2, anchor=NW, x=25, y=300)

    def Compare(self):

        self.frame3_1 = Frame(self.root)
        self.frame3_1.config(bg='#cfcdcc', height=680, width=1000)
        Label(self.frame3_1, text='frame1').place(in_=self.frame3_1, anchor=NW)
        self.frame3_1.place(x=0, y=0)
        Label(self.frame3_1,text='算法性能评估').place(in_=self.frame3_1,anchor=NW)
        self.Scenario_para()

        self.frame3_1.MT_pos=PIL.ImageTk.PhotoImage(PIL.Image.open('MT_pos.png').resize((250, 250)))
        picture=Label(self.frame3_1, image = self.frame3_1.MT_pos).place(in_=self.frame3_1, x=350, y=50, anchor=NW)

        self.frame3_1.Noise = PIL.ImageTk.PhotoImage(PIL.Image.open('Noise.png').resize((250, 250)))
        picture = Label(self.frame3_1, image=self.frame3_1.Noise).place(in_=self.frame3_1, x=650, y=50, anchor=NW)

        self.frame3_1.Heat = PIL.ImageTk.PhotoImage(PIL.Image.open('HeatMap.png').resize((250, 250)))
        picture = Label(self.frame3_1, image=self.frame3_1.Heat).place(in_=self.frame3_1, x=350, y=350, anchor=NW)

        self.frame3_1.Three_D = PIL.ImageTk.PhotoImage(PIL.Image.open('3D.png').resize((250, 250)))
        picture = Label(self.frame3_1, image=self.frame3_1.Three_D).place(in_=self.frame3_1, x=650, y=350, anchor=NW)

        #ana=analysis.Analysis(self.para)
        #ana.Change_MT_Pos()
        #ana.Change_Noise()

    def Heatmap(self):
        ana=analysis.Analysis(self.para)
        ana.Change_Noise()

    def plot3D(self):
        ana = analysis.Analysis(self.para)
        ana.losNlos()

def main():
    root = Tk()
    s=simulation(root)
    mainloop()
if __name__ == "__main__":
    main()
