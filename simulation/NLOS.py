from sympy import *
import numpy as np
import random
import math
import LOS

class Program:

    def __init__(self,bt):
        self.BT=bt
        self.BT_x = [b[0] for b in self.BT ]
        self.BT_y = [b[1] for b in self.BT]
        self.dis_ = [b[2] for b in self.BT]

    def LLS_NLOS(self):
        dis_, BT_x, BT_y = [b[2] for b in self.BT], [b[0] for b in self.BT], [b[1] for b in self.BT]
        A, p = [], []
        for bt_x, bt_y, dis in zip(BT_x[1:], BT_y[1:], dis_[1:]):
            A.append([bt_x - BT_x[0], bt_y - BT_y[0]])
            p.append([dis_[0] ** 2 - dis ** 2 - ((BT_x[0] ** 2 + BT_y[0] ** 2) - (bt_x ** 2 + bt_y ** 2))])
        A3 = np.array(A)
        p3 = np.array(p)
        pos = 0.5 * (np.linalg.inv((A3.T).dot(A3))).dot(A3.T).dot(p3)
        position = [round(pos[0][0], 1), round(pos[1][0], 1)]
        return position

    def target_fun(self,x_p,y_p,BT):
        dis, BT_x, BT_y = [b[2] for b in BT], [b[0] for b in BT], [b[1] for b in BT]
        target_fun=sum(list(map(lambda d,bt_x,bt_y:(d-((x_p-bt_x)**2+(y_p-bt_y)**2)**0.5)**2,dis,BT_x,BT_y)))
        return target_fun

    def lls(self,BT):
        dis_, BT_x, BT_y = [b[2] for b in BT], [b[0] for b in BT], [b[1] for b in BT]
        A, p = [], []
        for bt_x, bt_y, dis in zip(BT_x[1:], BT_y[1:], dis_[1:]):
            A.append([bt_x - BT_x[0], bt_y - BT_y[0]])
            p.append([dis_[0] ** 2 - dis ** 2 - ((BT_x[0] ** 2 + BT_y[0] ** 2) - (bt_x ** 2 + bt_y ** 2))])
        A3 = np.array(A)
        p3 = np.array(p)
        pos = 0.5 * (np.linalg.inv((A3.T).dot(A3))).dot(A3.T).dot(p3)
        position = [round(pos[0][0], 1), round(pos[1][0], 1)]
        return position

    def RSWG(self):
        target_BT=self.BT
        length = len(self.BT)
        x_min=self.lls(target_BT)
        Res=self.target_fun(x_min[0],x_min[1],target_BT)/length
        for i in range(0,length-3):
            temp_BT=target_BT
            for j in range(len(temp_BT)):
                temp=temp_BT[0:j]+temp_BT[j+1:]
                x_temp=self.lls(temp)
                res=self.target_fun(x_temp[0],x_temp[1],temp)/len(temp)
                if res<Res:
                    Res=res
                    target_BT=temp
                    target_X=x_temp
            if temp_BT==target_BT:
                break
        return target_BT,target_X


    def CRLB(self):
        print('crlb')


def Randd():
    BT=[]
    BT_Los_x, BT_Los_y, Dis_Los, MPE_Los, dis_Los = [], [], [], [], []
    BT_Nlos_x, BT_Nlos_y, Dis_Nlos, MPE_Nlos, dis_Nlos = [], [], [], [], []
    for i in range(8):
        x = random.random() * 1000
        y = random.random() * 680
        d = ((400 - x) ** 2 + (300 - y) ** 2) ** 0.5
        tag = random.randint(0, 1)
        if tag==0:
            BT_Los_x.append(x)
            BT_Los_y.append(y)
            Dis_Los.append(d)
            n = math.log(d) * random.gauss(0, 1)
            MPE_Los.append(n)
            dis_Los.append(d + n)
            BT.append([x,y,d+n])
        else:
            BT_Nlos_x.append(x)
            BT_Nlos_y.append(y)
            Dis_Nlos.append(d)
            n = math.log(d) * 3*random.expovariate(3)
            MPE_Nlos.append(n)
            dis_Nlos.append(d + n)
            BT.append([x,y,d+n])
    return BT


def main():
    BT=Randd()
    NLOS=Program(BT)
    NLOS.RSWG()

if __name__=="__main__":
    main()