import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import *
import cvxopt as cvx
import cvxpy as cp
import numpy as np
import random
import time
import math
# c=299552816


def plot_cicle(centers, rads, mt, bt_x, bt_y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for center, rad in zip(centers, rads):
        cir = plt.Circle((center[0], center[1]), radius=rad, color='y', fill=False)
        ax.add_patch(cir)

    ax.plot(bt_x, bt_y, 'ro')
    ax.plot(mt[0], mt[1], 'ro')
    plt.axis('scaled')
    plt.axis('equal')
    plt.show()


class Program:

    def __init__(self, bt):
        self.BT = bt
        self.BT_x = [b[0] for b in self.BT]
        self.BT_y = [b[1] for b in self.BT]
        self.dis_ = [b[2] for b in self.BT]
        self.number = len(bt)

# ----------------------------------------------For ML algorithm

    def RSR(self, dis, pos, BT_x,BT_y):
        sum = 0
        for d, p, bt_x,bt_y in zip(dis, pos, BT_x,BT_y):
            elem = (d - ((pos[0] - bt_x) ** 2 + (pos[1] - bt_y) ** 2) ** 0.5) ** 2
            sum = sum + elem
        return sum

    def cal(self,tag, paras, cords, c_p=0):
        sum, g_i = 0, []
        if tag == 0:
            for para, cord in zip(paras, cords):
                if para:
                    g_i.append((c_p - cord) / (2 * para ** 2))
                else:
                    g_i.append(1)
            return g_i

        if tag == 1:
            for para, cord in zip(paras, cords):
                sum = sum + para * cord
            return sum

    def cal1(self,s, P, X, Y, D):
        sum = 0
        for p, x, y, d in zip(P, X, Y, D):
            sum = sum + p * (s + x ** 2 + y ** 2 - d ** 2)
        return sum

    def cal2(self,s, P, X, Y, D):
        sum = 0
        if isinstance(s,complex):
            s=s.real
        else:
            s=abs(s)
        for p, x, y, d in zip(P, X, Y, D):
            sum = sum + p * (s + x ** 2 + y ** 2 - d ** 2)
        return sum

    def init_value(self):
        # deal with BT in a line
        if self.BT_y[0]==self.BT_y[1] and self.BT_y[0]==self.BT_y[2]:
            x=((self.dis_[1]**2-self.dis_[0]**2)-(self.BT_x[1]**2-self.BT_x[0]**2))/(2*(self.BT_x[0]-self.BT_x[1]))
            y=(self.dis_[0]**2-(x-self.BT_x[0])**2)**0.5+self.BT_y[0]

        elif self.BT_x[0]==self.BT_x[1] and self.BT_x[0]==self.BT_x[2]:
            y = ((self.dis_[1] ** 2 - self.dis_[0] ** 2) - (self.BT_y[1] ** 2 - self.BT_y[0] ** 2)) / (2 * (self.BT_y[0] - self.BT_y[1]))
            x = (self.dis_[0] ** 2 - (y - self.BT_y[0]) ** 2) ** 0.5 + self.BT_x[0]

        else:
            a,b=2*(self.BT_x[1]-self.BT_x[0]),2*(self.BT_y[1]-self.BT_y[0])
            c=(self.dis_[0]**2-self.dis_[1]**2)-((self.BT_x[0]**2-self.BT_x[1]**2)+(self.BT_y[0]**2-self.BT_y[1]**2))
            d,e=2*(self.BT_x[2]-self.BT_x[0]),2*(self.BT_y[2]-self.BT_y[0])
            f=(self.dis_[0] ** 2 - self.dis_[2] ** 2) - ((self.BT_x[0]**2 - self.BT_x[2]**2) + (self.BT_y[0]**2 - self.BT_y[2]**2))
            y = (f - d / a * c) / (e - d / a * b)
            x=(c-b*y)/a
        return x,y

    def estimate_one(self,x_ini,y_ini):
        self.g_i,self.h_i=self.cal(0,self.dis_,self.BT_x,c_p=x_ini),self.cal(0,self.dis_,self.BT_y,c_p=y_ini)
        a,b,c,d=self.cal(1,self.g_i,self.BT_x),self.cal(1,self.g_i,self.BT_y),self.cal(1,self.h_i,self.BT_x),self.cal(1,self.h_i,self.BT_y)
        A=2*np.array([[a,b],[c,d]])
        s=Symbol('s')
        e=self.cal1(s,self.g_i,self.BT_x,self.BT_y,self.dis_)
        f = self.cal1(s, self.h_i, self.BT_x, self.BT_y, self.dis_)
        B=np.array([e,f]).T
        try:
            A_=np.linalg.inv((A.T).dot(A)).dot(A.T)
        except:
            print('矩阵不可逆')
        s_=solve((A_[0][0]*e+A_[0][1]*f)**2+(A_[1][0]*e+A_[1][1]*f)**2-s,s)
        #print(s_)
        e_1=self.cal2(s_[0],self.g_i,self.BT_x,self.BT_y,self.dis_)
        f_1=self.cal2(s_[0],self.h_i,self.BT_x,self.BT_y,self.dis_)
        B_1= np.array([e_1, f_1]).T

        e_2 = self.cal2(s_[1], self.g_i, self.BT_x, self.BT_y, self.dis_)
        f_2 = self.cal2(s_[1], self.h_i, self.BT_x, self.BT_y, self.dis_)
        B_2 = np.array([e_2, f_2]).T
        try:
            position1=np.linalg.inv(A).dot(B_1)
            position2 = np.linalg.inv(A).dot(B_2)
            t1=self.RSR(self.dis_,position1,self.BT_x,self.BT_y)
            t2=self.RSR(self.dis_,position2,self.BT_x,self.BT_y)
            return position1 if t1<t2 else position2
        except:
            print("矩阵不可逆")

    def estimate(self):
        x_ini,y_ini=self.init_value()
        pos_,J_v = [],[]
        location = [x_ini, y_ini]
        for i in range(5):
            sum=0
            for d,cord_x,cord_y in zip(self.dis_,self.BT_x,self.BT_y):
                sum=sum+(d-((location[0]-cord_x)**2+(location[1]-cord_y)**2)**0.5)**2
            J_v.append(sum)
            location = self.estimate_one(float(location[0]), float(location[1]))
            pos_.append(list(location))
        id = J_v.index(min(J_v))
        #print("estimate_location:",pos_[id])
        pos_[id]=[round(pos_[id][0],1),round(pos_[id][1],1)]
        return pos_[id]

#-------------------------------------------------------------------------for LS

    def LLS_1_E(self):
        A,p=[],[]
        for bt_x,bt_y,dis in zip(self.BT_x[1:],self.BT_y[1:],self.dis_[1:]):
            A.append([bt_x-self.BT_x[0],bt_y-self.BT_y[0]])
            p.append([self.dis_[0]**2-dis**2-((self.BT_x[0]**2+self.BT_y[0]**2)-(bt_x**2+bt_y**2))])
        A3=np.array(A)
        p3=np.array(p)
        pos=0.5*(np.linalg.inv((A3.T).dot(A3))).dot(A3.T).dot(p3)
        position=[round(pos[0][0],1),round(pos[1][0],1)]
        return position

    def LLS_AVE_E(self):
        A,p=[],[]
        d_ave=sum(list(map(lambda d: d**2,self.dis_)))/len(self.dis_)
        k_ave=sum(list(map(lambda x,y: x**2+y**2,self.BT_x,self.BT_y)))/len(self.BT_x)
        x_ave=sum(self.BT_x)/len(self.BT_x)
        y_ave = sum(self.BT_y) / len(self.BT_y)
        for bt_x,bt_y,dis in zip(self.BT_x,self.BT_y,self.dis_):
            A.append([bt_x-x_ave,bt_y-y_ave])
            p.append([d_ave-dis**2+(bt_x**2+bt_y**2)-k_ave])
        A3=np.array(A)
        p3=np.array(p)
        pos=0.5*(np.linalg.inv((A3.T).dot(A3))).dot(A3.T).dot(p3)
        position=[round(pos[0][0],1),round(pos[1][0],1)]
        return position

    def LLS_RS_E(self):
        A,p=[],[]
        i_d=self.dis_.index(min(self.dis_))
        d_min,x_min,y_min=self.dis_[i_d],self.BT_x[i_d],self.BT_y[i_d]
        BT_X=self.BT_x[0:i_d]+self.BT_x[i_d+1:]
        BT_Y = self.BT_y[0:i_d] + self.BT_y[i_d + 1:]
        dis_=self.dis_[0:i_d]+self.dis_[i_d+1:]
        for bt_x,bt_y,dis in zip(BT_X,BT_Y,dis_):
            A.append([bt_x-x_min,bt_y-y_min])
            p.append([d_min**2-dis**2-((x_min**2+y_min**2)-(bt_x**2+bt_y**2))])
        A3=np.array(A)
        p3=np.array(p)
        pos=0.5*(np.linalg.inv((A3.T).dot(A3))).dot(A3.T).dot(p3)
        position=[round(pos[0][0],1),round(pos[1][0],1)]
        return position

    def CWLS(self):
        B_ = self.dis_
        B = np.diag(B_)

    def WLS(self):
        B_ = [1/(d ** 2) for d in self.dis_]
        B = np.diag(B_)

#------------------------------------------------------------------------------------SDR
    def R_LS_E(self):
        num=len(self.dis_)
        X, G = cp.Variable((3, 3),PSD=true), cp.Variable((num + 1, num + 1),PSD=true)# PSD=true means semidefinite
        expr=0
        constraints = [G[num, num] == 1, X[2, 2] == 1]
        for i in range(num):
            C_i=[[1,0,-self.BT_x[i]],[0,1,-self.BT_y[i]],[-self.BT_x[i],-self.BT_y[i],(self.BT_x[i]**2+self.BT_y[i]**2)]]
            expr=expr+G[i,i]-2*self.dis_[i]*G[num,i]+self.dis_[i]**2
            constraints += [G[i, i] == cp.trace(C_i@X)]
        obj = cp.Minimize(expr)
        prob = cp.Problem(obj,constraints)
        prob.solve()
        position = [round(X.value[0,2],1), round(X.value[1,2], 1)]
        #print(position)
        return position

    def kesi(self,lamda,A,D,b,f):
        kesi_lamda = 0
        if np.linalg.det((A.T).dot(A) + lamda*D) == 0:
            kesi_lamda = np.inf
        else:
            y_lamda = np.linalg.solve(A.T.dot(A) + lamda * D, A.T.dot(b)  - lamda * f)
            kesi_lamda = y_lamda.T.dot(D).dot(y_lamda) + 2 * (f.T) .dot(y_lamda)
        return y_lamda,kesi_lamda

    def SR_LS(self):
        A = np.array([[-2 * x, -2 * y, 1] for x, y in zip(self.BT_x, self.BT_y)])
        D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        b = np.array([d ** 2 - (x ** 2 + y ** 2) for x, y, d in zip(self.BT_x, self.BT_y, self.dis_)])
        f = np.array([0,0,-0.5])
        A_T_A = (A.T).dot(A)
        A_T_A_inv = np.linalg.inv(A_T_A)
        eig = list(np.linalg.eig(A_T_A_inv)[0])
        A_T_A_eig = np.diag(eig)
        #print("A_T_A_eig:", A_T_A_eig)
        A_T_A_mat = np.linalg.eig(A_T_A_inv)[1].T
        #print("A_T_A_mat:", A_T_A_mat)
        square_eig = [x ** 0.5 for x in eig]
        # print("square_eig",square_eig)
        square_eig_mat = np.diag(square_eig)
        # print("square_mat", square_eig_mat)
        ATA_sq = A_T_A_mat.dot(square_eig_mat).dot(A_T_A_mat)
        lamda_array = list(np.linalg.eig(ATA_sq.dot(D).dot(ATA_sq))[0])
        #print("lamda_array:", lamda_array)
        lamda1 = max(lamda_array)
        if lamda1 ==0:
            lamda1 = 0.1
        #print("lamda1:", lamda1)
        left = -1 / lamda1
        left_kesi = np.inf
        right, err = 10, 0.1
        _, right_kesi = self.kesi(right, A, D, b, f)
        #print("right_kesi:", kesi)
        while right_kesi>=0:
            left = right
            left_kesi = right_kesi
            right=10*right
            _, right_kesi = self.kesi(right, A, D, b, f)
        mid = (left+right)/2
        #print("right_kesi:", right_kesi)
        #print("left_kesi:",left_kesi)
        #print("origin_left:",left)
        while right-left > 0.01:
            if np.isfinite(left_kesi):
                p_rf = left + left_kesi * (right - left) / (left_kesi - right_kesi)
                # interval midpoint
                p_mid = (left + right) / 2
                # weighted average of the midpoint and regula falsi point
                bsfac = 0.95
                mid = bsfac * p_rf + (1 - bsfac) * p_mid#试位法
            else:
                mid = (right+left)/2
            _, kesi = self.kesi(mid, A, D, b, f)
            if kesi < -err:
                right = mid
                mid = (left + right) / 2
            elif kesi > err:
                left = mid
                mid = (left + right) / 2
            else:
                break
        y,kesi_lamda = self.kesi(mid,A,D,b,f)
        pos = [round(abs(y[0]),1),round(abs(y[1]),1)]
        #print("pos:",pos)
        return pos

#-----------------------------------------------------NewTon-Gaussian

    def Grad(self,f,x1,x2,x_i,y_i):
        f1=diff(f,x1)
        f2=diff(f,x2)
        grad=np.array([[f1.subs([(x1,x_i),(x2,y_i)])],[f2.subs([(x1,x_i),(x2,y_i)])]])
        return grad

    def Hess(self,f,x1,x2,x_i,y_i):
        f1=diff(f,x1)
        f2=diff(f,x2)
        f11,f12,f21,f22=diff(f1,x1),diff(f1,x2),diff(f2,x1),diff(f2,x2)
        hess=np.array([[f11.subs([(x1,x_i),(x2,y_i)]),f12.subs([(x1,x_i),(x2,y_i)])],
                       [f21.subs([(x1,x_i),(x2,y_i)]),f22.subs([(x1,x_i),(x2,y_i)])]])
        hess = np.array(hess, dtype='float')
        return hess

    def NewTon_Gauss(self):
        err,count=0.01,0
        X=self.init_value()
        x_k,x_k1=np.array([[X[0]],[X[1]]]),np.array([[0],[0]])
        x_p,y_p=symbols('x,y')
        target_fun=sum(list(map(lambda d,bt_x,bt_y:(d-((x_p-bt_x)**2+(y_p-bt_y)**2)**0.5)**2,self.dis_,self.BT_x,self.BT_y)))
        pos=[]
        while True:
            #print(x_k)
            pos.append([round(x_k[0][0],1),round(x_k[1][0],1)])
            x_err=x_k-x_k1
            if x_err[0][0]**2+x_err[1][0]**2<=err**2:
                break
            grad = self.Grad(target_fun, x_p, y_p, x_k[0][0], x_k[1][0])
            hess = self.Hess(target_fun, x_p, y_p, x_k[0][0], x_k[1][0])
            if grad[0][0]**2+grad[1][0]**2<err*2:
                break
            x_k1=x_k
            x_k=x_k-(np.linalg.inv(hess)).dot(grad)
            count=count+1
            if count>10:
                break
        return pos[-1]

    def SAA(self):
        X_target = [1000,600]
        E_1 = sum([self.dis_[i]-((self.BT_x[i]-X_target[0])**2+(self.BT_y[i]-X_target[1])**2)**0.5 for i in range(len(self.BT_x))])
        X = [(0.9+random.random()/10)*X_target[0], (0.9+random.random(0,0.1)/10)*X_target[1]]
        E_2 = sum([self.dis_[i]-((self.BT_x[i]-X[0])**2+(self.BT_y[i]-X[1])**2)**0.5 for i in range(len(self.BT_x))])
        if E_2<E_1:
            X_target = X
        else:
            math.exp(-(E_2-E_1)/1)
#---------------------------------------------------------------------------------

    def CRLB_LOS(self,MT_x,MT_y,sigma):
        a, b, c = 0, 0, 0
        for k in range(len(self.BT_x)):
            deno = sigma ** 2 * ((MT_x - self.BT_x[k]) ** 2 + (MT_y - self.BT_y[k]) ** 2)
            a = a + (MT_x - self.BT_x[k]) ** 2 / deno
            b = b + ((MT_x - self.BT_x[k]) * (MT_y - self.BT_y[k])) / deno
            c = c + (MT_y - self.BT_y[k]) ** 2 / deno
        try:
            Fisher = np.array([[a, b], [b, c]])
            crlb = 1 / (Fisher)
            #print(crlb)
        except:
            print("矩阵不可逆")
        # datas=list(np.array(datas)[10:40,10:40])
        return crlb[0,0],crlb[1,1]
class Program_t:

    def __init__(self,bt):
        self.BT = bt
        self.BT_x = [b[0] for b in self.BT]
        self.BT_y = [b[1] for b in self.BT]
        self.T=[b[2] for b in self.BT]
        self.number = len(bt)

    def lsT(self):
        A = []
        b = []
        x1, y1, d1 = self.BT_x[0], self.BT_y[0], self.T[0]
        for i in range(1,self.number):
            A.append([2*(self.BT_x[i]-x1), 2*(self.BT_y[i]-y1), 2*(d1-self.T[i])])
            b.append(self.BT_x[i]**2-x1**2+self.BT_y[i]**2-y1**2+d1**2-self.T[i]**2)
        A = np.array(A)
        b = np.array(b)
        pos = (np.linalg.inv((A.T).dot(A))).dot(A.T).dot(b)
        position = [round(pos[0], 1), round(pos[1], 1)]
        print(pos)
        return position

    def LS_steps(self):
        num = len(self.T)
        up_control = 2*max(self.T)**2
        Q = cp.Variable((num,num))
        tao = cp.Variable((num,1))
        y_ = cp.Variable((2,1))
        y_s = cp.Variable((1,1))
        yita = 0.00005*sum(self.T) / num
        G = np.eye(num)-np.ones((num,num))
        t = np.array([self.T]).T
        expr1 = cp.trace((cp.transpose(G)) @ G @ (Q- cp.multiply(2,t @ (cp.transpose(tao)))+t @ (cp.transpose(t))))
        expr2 = yita*cp.sum(Q)
        expr = expr1+ expr2
        Q_ = cp.bmat([[Q,tao],[cp.transpose(tao),[[1]]]])
        Y = cp.bmat([[np.eye(2),y_],[cp.transpose(y_),y_s]])
        constraints = [Q_ >> 0, Y >> 0, cp.max(Q)<=up_control]
        for i in range(num):
            X = np.array([self.BT_x[i],self.BT_y[i],-1]).T
            constraints += [Q[i, i] == cp.transpose(X) @ Y @ X]
            for j in range(i+1,num):
                X_j = np.array([self.BT_x[j], self.BT_y[j],-1]).T
                constraints += [Q[i, j] >= cp.abs(cp.transpose(X) @ Y @ X_j)]
        obj = cp.Minimize(expr)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        position = y_.value
        #print(expr1.value)
        #print(expr2.value)
        #print(prob.value)
        #print(prob.status)
        print(position)
        return position


#------------------------------------------------------------------------画圆函数

    def plot_cicle(centers, rads, MT, BT_x, BT_y):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for center, rad in zip(centers, rads):
            cir = plt.Circle((center[0], center[1]), radius=rad, color='y', fill=False)
            ax.add_patch(cir)

        ax.plot(BT_x, BT_y, 'ro')
        ax.plot(MT[0], MT[1], 'ro')
        plt.axis('scaled')
        plt.axis('equal')
        plt.show()

def exe_time():
    BT_x, BT_y, Dis, MPE, dis_ = [], [], [], [], []
    ave, sigma = 0, 2
    MT = [random.random()*600+200, random.random()*400+100]
    bt = []
    number = random.randint(3,8)
    for i in range(number):
        x = random.random() * 800
        BT_x.append(x)
        y = random.random() * 500
        BT_y.append(y)
        d = ((MT[0] - BT_x[i]) ** 2 + (MT[1] - BT_y[i]) ** 2) ** 0.5
        Dis.append(d)
        n = random.gauss(ave, sigma)
        dis_.append(d + n)
        bt.append((x, y, d + n))

    #los = Program_t(bt)

    #start = time.time()
    #los.LS_steps()
    #end = time.time()
    #lsteps_exe = end - start
    los = Program(bt)

    start = time.time()
    los.estimate()
    end = time.time()
    aml_exe = end-start

    start = time.time()
    los.LLS_1_E()
    end = time.time()
    lls_1_exe = end - start

    start = time.time()
    los.NewTon_Gauss()
    end = time.time()
    N_G_exe = end - start

    start = time.time()
    los.SR_LS()
    end = time.time()
    sr_ls_exe = end - start

    start = time.time()
    los.R_LS_E()
    end = time.time()
    r_ls_exe = end - start

    print("aml:",aml_exe)
    print("lls:",lls_1_exe)
    print("NG:",N_G_exe)
    print("sr_ls:",sr_ls_exe)
    print("r_ls:",r_ls_exe)
    #return aml_exe,lls_1_exe,N_G_exe,sr_ls_exe,r_ls_exe
    bias = [aml_exe,lls_1_exe,N_G_exe,sr_ls_exe,r_ls_exe]
    name=['AML','LLS','NG','SR-LS','R-LS']
    plt.bar(range(len(bias)), bias, tick_label=name)
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time/s')
    plt.title('Execution Time')
    plt.show()

def main_():
    aml, lls_1, N_G, sr_ls, r_ls=[],[],[],[],[]
    for i in range(30):
        aml_exe,lls_1_exe,N_G_exe,sr_ls_exe,r_ls_exe = exe_time()
        aml.append(aml_exe)
        lls_1.append(lls_1_exe)
        N_G.append(N_G_exe)
        sr_ls.append(sr_ls_exe)
        r_ls.append(r_ls_exe)
        print(i)
    aml.pop(aml.index(max(aml)))
    lls_1.pop(lls_1.index(max(lls_1)))
    N_G.pop(N_G.index(max(N_G)))
    sr_ls.pop(sr_ls.index(max(sr_ls)))
    r_ls.pop(r_ls.index(max(r_ls)))
    bias = [sum(aml) / 29, sum(lls_1) / 29, sum(N_G) / 29, sum(sr_ls) / 29, sum(r_ls)/29]
    name = ['AML', 'LLS1', 'N_G', 'SR_LS', 'R_LS']
    plt.bar(range(len(bias)), bias, tick_label=name)
    plt.xlabel('Algorihm')
    plt.ylabel('Exe_time/s')
    plt.title('Exe_time')
    plt.show()
    #sdr.test()

def CRLB_LOS():
    #BT_x=[250,250,750,750]
    #BT_y=[250,750,250,750]
    BT_x=[int(500+250*math.cos(i*math.pi/4)) for i in range(8)]
    BT_y=[int(500+250*math.sin(i*math.pi/4)) for i in range(8)]
    print(BT_x,BT_y)
    MT_x,MT_y=0,0
    sigma = 5
    datas1, datas2 = [], []
    for i in range(100):
        print(i)
        data1, data2 = [], []
        MT_x = 10 * i
        for j in range(100):
            MT_y = 10 * j
            a, b, c = 0, 0, 0
            for k in range(len(BT_x)):
                deno = sigma ** 2 * ((MT_x - BT_x[k]) ** 2 + (MT_y - BT_y[k]) ** 2)
                if deno:
                    a = a + (MT_x - BT_x[k]) ** 2 / deno
                    b = b + ((MT_x - BT_x[k]) * (MT_y - BT_y[k])) / deno
                    c = c + (MT_y - BT_y[k]) ** 2 / deno
                else:
                    a,b,c=0.1,0.1,0.1
                    break
            try:
                Fisher = np.array([[a, b], [b, c]])
                crlb = 1 / (Fisher)
                #crlb=np.linalg.inv(Fisher)
                # print(crlb)
            except:
                print("矩阵不可逆")
            # datas=list(np.array(datas)[10:40,10:40])
            #return crlb[0, 0], crlb[1, 1]
            #noise1 = (crlb[0, 0]+crlb[1, 1])**0.5
            noise1 = (crlb[0, 0] + crlb[1, 1]) ** 0.5/sigma
            #print(noise1)
            data1.append(float(noise1))
        datas1.append(data1)
        #print(datas1)

    x_tick = [100 * i for i in range(10)]
    y_tick = [100 * i for i in range(10)]
    ax1 = plt.subplot()
    for l in range(len(BT_x)):
        #plt.scatter(BT_x[l], BT_y[l],s=y,  # 设置散点x大小color='r',  # 设置散点颜色(也可以是一个列表，控制每个点的颜色)marker="o",  # 设置散点形状linewidths=3,  # 设置边框的宽度edgecolors='r',  # 设置边框的颜色alpha=0.9,  # 设置透明度 )
        #plt.Rectangle([50, 50], 20,20,color='r',fill=True)
        circle = plt.Circle((BT_x[l]/10, BT_y[l]/10), 1, color='y', fill=True)
        plt.gcf().gca().add_artist(circle)
    #    ax1.add_patch(rec)
    im1 = ax1.imshow(datas1, cmap=cm.binary)
    #plt.text(50, 50, 'FT', color='red',fontsize=5)
    #print(im1)
    ax1.set_xticks(np.arange(5, 100, 10), labels=x_tick)
    ax1.set_yticks(np.arange(5, 100, 10), labels=y_tick)

    plt.title('GDOP Heat Map')
    plt.xlabel('X/m')
    plt.ylabel('Y/m')
    cbar = plt.colorbar(im1)
    cbar.set_label('RMSE/m')
    plt.show()
def main():
    bt_x = [0,0,800,800]
    bt_y = [0,600,0,600]
    bt = []
    stepbias,lsbias,count =0,0,3
    for i in range(count):
        d0 = random.random() * 50
        mt = [random.random() *800, random.random()*600]
        for x, y in zip(bt_x,bt_y):
            d = ((x - mt[0])**2+(y-mt[1])**2)**0.5+d0+random.gauss(0,15)
            bt.append([x, y, d])
        item = Program_t(bt)
        #print(d0)
        lsstep_pos=item.LS_steps()
        print("lsstep_pos:",lsstep_pos)
        lspos=item.lsT()
        stepbias+=((lsstep_pos[0][0] - mt[0]) ** 2 + (lsstep_pos[1][0] - mt[1]) ** 2) ** 0.5
        lsbias+=((lspos[0] - mt[0]) ** 2 + (lspos[1] - mt[1]) ** 2) ** 0.5
    bias = [stepbias/count,lsbias/count]
    name = ['LS-Step','LLS']
    plt.bar(range(len(bias)), bias, tick_label=name)
    plt.xlabel('Algorihm')
    plt.ylabel('Average Bias/m²')
    plt.title('SDP And LLS')
    plt.show()
if __name__=="__main__":
    #CRLB_LOS()
    exe_time()