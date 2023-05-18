from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
import numpy as np
import random
import math
import NLOS
import LOS

class Analysis:

    def __init__(self, para):
        self.para={}
        self.para['BT'] = para['BT']
        self.para['BT_LOS'] = para['BT_LOS']
        self.para['BT_NLOS'] = para['BT_NLOS']
        self.para['lamda'], self.para['ave'], self.para['sigma'] = para['lamda'], para['ave'], para['sigma']
        self.para['MT_x'], self.para['MT_y'] = para['MT_x'], para['MT_y']

    def update(self):
        aml, lls1, lls_ave, lls_rs, ng, r_ls, lls_nlos, rwgh = 0,0,0,0,0,0,0,0
        Algo_Noise = {}
        for i, bt_para in enumerate(self.para['BT_LOS']):
            d=((self.para['MT_x'] - bt_para[0]) ** 2 + (self.para['MT_y'] - bt_para[1]) ** 2) ** 0.5
            n=math.log(d) * random.gauss(self.para['ave'], self.para['sigma'])
            self.para['BT_LOS'][i][2]=d+n

        '''
        for i, bt_para in enumerate(self.para['BT_NLOS']):
            d = ((self.para['MT_x'] - bt_para[0]) ** 2 + (self.para['MT_y'] - bt_para[1]) ** 2) ** 0.5
            n = math.log(d) * (self.para['lamda'] * np.random.exponential(self.para['lamda']) + random.gauss(self.para['ave'],self.para['sigma']))
            self.para['BT_NLOS'][i][2] = d + n

        self.para['BT'] = self.para['BT_LOS'] + self.para['BT_NLOS']
        '''

        instance = LOS.Program(self.para['BT_LOS'])

        aml_x, aml_y = instance.estimate()
        aml = ((aml_x - self.para['MT_x']) ** 2 + (aml_y - self.para['MT_y']) ** 2) ** 0.5

        lls1_x, lls1_y = instance.LLS_1_E()
        lls = ((lls1_x - self.para['MT_x']) ** 2 + (lls1_y - self.para['MT_y']) ** 2) ** 0.5

        llsave_x, llsave_y = instance.LLS_AVE_E()
        lls_ave = ((llsave_x - self.para['MT_x']) ** 2 + (llsave_y - self.para['MT_y']) ** 2) ** 0.5

        llsrs_x, llsrs_y = instance.LLS_RS_E()
        lls_rs = ((llsrs_x - self.para['MT_x']) ** 2 + (llsrs_y - self.para['MT_y']) ** 2) ** 0.5

        sr_ls_x,sr_ls_y=instance.SR_LS()
        sr_ls = ((sr_ls_x - self.para['MT_x']) ** 2 + (sr_ls_y - self.para['MT_y']) ** 2) ** 0.5

        #r_ls_instance = LOS.Program(self.para['BT_LOS'])
        #r_ls_x, r_ls_y = r_ls_instance.R_LS_E()
        #r_ls = ((r_ls_x - self.para['MT_x']) ** 2 + (r_ls_y - self.para['MT_y']) ** 2) ** 0.5
        #lls_aver_instance = LOS.Program(self.para['BT_LOS'])
        #lls_ave_x, lls_ave_y = lls_aver_instance.LLS_AVE_E()
        #lls_ave = ((lls_ave_x - self.para['MT_x']) ** 2 + (lls_ave_y - self.para['MT_y']) ** 2) ** 0.5
        #lls_rs_instance = LOS.Program(self.para['BT_LOS'])
        #lls_rs_x, lls_rs_y = lls_rs_instance.LLS_RS_E()
        #lls_rs = ((lls_rs_x - self.para['MT_x']) ** 2 + (lls_rs_y - self.para['MT_y']) ** 2) ** 0.5
        ng_x, ng_y=  instance.NewTon_Gauss()
        ng = ((ng_x - self.para['MT_x']) ** 2 + (ng_y - self.para['MT_y']) ** 2) ** 0.5

        CRLB = LOS.Program(self.para['BT_LOS'])
        crlb_x, crlb_y = CRLB.CRLB_LOS(self.para['MT_x'],self.para['MT_y'],self.para['sigma'])
        crlb=(crlb_x+crlb_y)**0.5
        #s_ls_instance=LOS.Program(self.para['BT_LOS'])
        #r_ls_x, r_ls_y = s_ls_instance.R_LS_E()
        #r_ls = (r_ls_x-self.para['MT_x'])**2+(r_ls_y-self.para['MT_y'])**2
        '''
        lls_nlos_instance = NLOS.Program(self.para['BT'])
        lls_nlos_x, lls_nlos_y = lls_nlos_instance.LLS_NLOS()
        lls_nlos = ((lls_nlos_x - self.para['MT_x']) ** 2 + (lls_nlos_y - self.para['MT_y']) ** 2) ** 0.5
        rwgh_instance = NLOS.Program(self.para['BT'])
        NLOS_pos, rwgh_pos = rwgh_instance.RSWG()
        rwgh_x, rwgh_y = rwgh_pos[0], rwgh_pos[1]
        rwgh = ((rwgh_x - self.para['MT_x']) ** 2 + (rwgh_y - self.para['MT_y']) ** 2) ** 0.5
        '''
        Algo_Noise = {'AML': aml, 'LLS':lls,'Newton': ng,'SR_LS':sr_ls,'CRLB':crlb,'LLS_AVE':lls_ave,'LLS_RS':lls_rs}
        return Algo_Noise

    def Change_Noise(self):
        aml, lls1,lls_ave,lls_rs,sr_ls, r_ls, ng,crlb, sigma =[],[],[],[], [], [], [], [], []
        count, skip,time = 20, 1,10
        for i in range(time):
            print('1')
            self.para['sigma'] = 2+skip * i
            sigma.append(2+skip * i)
            aml1, lls11, sr_ls1, r_ls1,crlb1, lls_ave1,lls_rs1,ng1=[],[],[], [], [], [], [], []
            for i in range(count):
                x = random.random() * 450+200
                y = random.random() * 450+50
                self.para['MT_x'], self.para['MT_y'] = x, y
                Noise = self.update()
                aml1.append(Noise['AML'])
                lls11.append(Noise['LLS'])
                sr_ls1.append(Noise['SR_LS'])
                #r_ls1.append(Noise['R_LS'])
                lls_ave1.append(Noise['LLS_AVE'])
                lls_rs1.append(Noise['LLS_RS'])
                ng1.append(Noise['Newton'])
                crlb1.append(Noise['CRLB'])
            #print(sr_ls1)
            bias = {'LLS_ave':sum(lls_ave1)/count,'LLS_rs':sum(lls_rs1)/count,'CRLB':sum(crlb1)/count,'AML': sum(aml1) / count, 'LLS': sum(lls11) / count,'R_LS':sum(r_ls1)/count, 'SR_LS': sum(sr_ls1) / count, 'Newton': sum(ng1) / count}
            aml.append(bias['AML'])
            lls1.append(bias['LLS'])
            lls_rs.append(bias['LLS_rs'])
            lls_ave.append(bias['LLS_ave'])
            sr_ls.append(bias['SR_LS'])
            #r_ls.append(bias['R_LS'])
            ng.append(bias['Newton'])
            crlb.append(bias['CRLB'])

        print(aml,lls1,sr_ls,r_ls,ng,crlb)
        plt.plot(sigma, aml, label='AML')
        plt.plot(sigma, lls1, label='LLS1')
        plt.plot(sigma, lls_ave, label='LLS_Ave')
        plt.plot(sigma, lls_rs, label='LLS_RS')
        plt.plot(sigma, sr_ls, label='SR_LS')
        #plt.plot(sigma, r_ls, label='R_LS')
        plt.plot(sigma, ng, label='Newton')
        plt.plot(sigma, crlb, label='CRLB')
        plt.xlabel('Gauss_Noise Var/m')
        plt.ylabel('Average_Bias/m')
        plt.title('Change Noise')
        plt.legend()
        plt.show()

    def Change_MT_Pos(self):
        aml, lls1, lls_ave, lls_rs, ng, r_ls, lls_nlos, rwgh = [], [], [], [], [], [], [], []
        count = 10
        for i in range(count):
            print('1')
            x = random.random() * 800
            y = random.random() * 600
            self.para['MT_x'], self.para['MT_y'] = x, y
            Noise = self.update()
            aml.append(Noise['AML'])
            lls1.append(Noise['LLS1'])
            lls_ave.append(Noise['LLS_AVE'])
            lls_rs.append(Noise['LLS_RS'])
            ng.append(Noise['Newton'])
            #r_ls.append(Noise['R_LS'])
            #lls_nlos.append(Noise['LLS_NLOS'])
            #rwgh.append(Noise['RWGH'])

        bias = [sum(aml) / count, sum(lls1) / count, sum(lls_ave) / count, sum(lls_rs) / count, sum(ng) / count]
        name = ['AML', 'LLS1', 'LLS_AVE', 'LLS_RS', 'N_G']
        plt.bar(range(len(bias)), bias, tick_label=name)
        plt.xlabel('Algorihm')
        plt.ylabel('Average_Bias')
        plt.title('Change MT')
        plt.show()

    def Bt_pos(self):
        print('hello')

    def Heatmap(self):
        datas1, datas2 = [], []
        for i in range(50):
            print(i)
            data1, data2 = [], []
            self.para['MT_x'] = 20*i
            for j in range(50):
                self.para['MT_y'] =20*j
                for k, bt_para in enumerate(self.para['BT_LOS']):
                    d = ((self.para['MT_x'] - bt_para[0]) ** 2 + (self.para['MT_y'] - bt_para[1]) ** 2) ** 0.5
                    n = math.log(d+1) * random.gauss(self.para['ave'], self.para['sigma'])
                    self.para['BT_LOS'][k][2] = d + n
                lls1_instance = LOS.Program(self.para['BT_LOS'])

                lls1_x, lls1_y = lls1_instance.estimate()
                noise1 = ((lls1_x - self.para['MT_x']) ** 2 + (lls1_y - self.para['MT_y']) ** 2)**0.5
                #noise1 = (lls1_x + lls1_y)
                #print(noise1)
                data1.append(float(noise1))
            datas1.append(data1)
            #print(datas1)
        x_tick=[100*i for i in range(10)]
        y_tick=[100*i for i in range(10)]
        ax1=plt.subplot()
        for l in range(len(self.para['BT_LOS'])):
            # plt.scatter(BT_x[l], BT_y[l],s=y,  # 设置散点x大小color='r',  # 设置散点颜色(也可以是一个列表，控制每个点的颜色)marker="o",  # 设置散点形状linewidths=3,  # 设置边框的宽度edgecolors='r',  # 设置边框的颜色alpha=0.9,  # 设置透明度 )
            # plt.Rectangle([50, 50], 20,20,color='r',fill=True)
            circle = plt.Circle((self.para['BT_LOS'][l][0] / 20, self.para['BT_LOS'][l][1] / 20), 1, color='y', fill=True)
            plt.gcf().gca().add_artist(circle)
        im1=ax1.imshow(datas1,cmap=cm.binary)
        ax1.set_xticks(np.arange(2,50,5), labels=x_tick)
        ax1.set_yticks(np.arange(2,50,5), labels=y_tick)

        plt.title('AML Error Heat Map')
        plt.xlabel('X/m')
        plt.ylabel('Y/m')
        cbar=plt.colorbar(im1)
        cbar.set_label('RMSE/m')
        plt.show()

    def plot3D(self):
        figure = plt.figure()
        ax = Axes3D(figure)
        X = np.arange(0, 1000, 10)
        Y = np.arange(0, 600, 10)
        X, Y = np.meshgrid(X, Y)
        Z = sum([(self.para['BT_LOS'][i][2]-((self.para['BT_LOS'][i][0]-X)**2+(self.para['BT_LOS'][i][1]- Y)**2)**0.5)**2 for i in range(len(self.para['BT_LOS']))])
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        ax.text(self.para['MT_x'],self.para['MT_y'],15,"MT")
        plt.title('3D')
        plt.xlabel('X/m')
        plt.ylabel('Y/m')
        plt.show()

    def losNlos(self):
        bias = []
        losNoise,nlosNoise=0,0
        name = ['LOS','NLOS']
        count = 100
        for i in range(count):
            print('1')
            x = random.random() * 800
            y = random.random() * 600
            self.para['MT_x'], self.para['MT_y'] = x, y
            for i, bt_para in enumerate(self.para['BT_LOS']):
                d = ((self.para['MT_x'] - bt_para[0]) ** 2 + (self.para['MT_y'] - bt_para[1]) ** 2) ** 0.5
                n = math.log(d) * random.gauss(self.para['ave'], self.para['sigma'])
                self.para['BT_LOS'][i][2] = d + n

            for i, bt_para in enumerate(self.para['BT_NLOS']):
                d = ((self.para['MT_x'] - bt_para[0]) ** 2 + (self.para['MT_y'] - bt_para[1]) ** 2) ** 0.5
                n = math.log(d) * (self.para['lamda'] * np.random.exponential(self.para['lamda']) + random.gauss(
                    self.para['ave'], self.para['sigma']))
                self.para['BT_NLOS'][i][2] = d + n

            self.para['BT'] = self.para['BT_LOS'] + self.para['BT_NLOS']
            lls = LOS.Program(self.para['BT_LOS'])
            nlos = NLOS.Program(self.para['BT'])
            lpos = lls.LLS_1_E()
            losn = ((lpos[0] - self.para['MT_x']) ** 2 + (lpos[1] - self.para['MT_y']) ** 2) ** 0.5
            npos = nlos.LLS_NLOS()
            nlosn = ((npos[0] - self.para['MT_x']) ** 2 + (npos[1] - self.para['MT_y']) ** 2) ** 0.5
            losNoise += losn
            nlosNoise += nlosn
        bias = [losNoise/count,nlosNoise/count]
        plt.bar(range(len(bias)), bias, tick_label=name)
        plt.xlabel('Scenario')
        plt.ylabel('Average_Bias/m')
        plt.title('LLS for LOS and NLOS')
        plt.show()

def distribution():
    lls_Noise,Aml_noise, SR_LS_noise= [],[],[]
    for time in range(2):
        print(time)
        Num = 8
        lls_noise,aml_noise,sr_ls_noise=[],[],[]
        for count in range(10000):
            print(count)
            BT_LOS_c,BT_LOS_s=[], []
            theta=random.random()*2*math.pi
            #MT_x,MT_y=random.random()*1200,random.random()*1200
            if time==0:
                MT_x, MT_y =400+math.sin(theta) * 400*random.random(),400 + math.cos(theta) * 400*random.random()
            else:
                MT_x, MT_y = 800 + abs(math.sin(theta)) * 400 * random.random(), 800 + abs(math.cos(theta)) * 400 * random.random()
            for i in range(Num):
                x = 400+math.sin(2 * math.pi / Num * i) * 400
                y = 400 + math.cos(2 * math.pi / Num * i) *400
                d = ((MT_x - x) ** 2 + (MT_y - y) ** 2) ** 0.5
                n = random.gauss(0, 10)
                BT_LOS_c.append([x,y,d+n])
            '''
            d0=((MT_x-200)  ** 2 + (MT_y-200) ** 2) ** 0.5
            d1=((MT_x-200)  ** 2 + (MT_y - 600) ** 2) ** 0.5
            d2=((MT_x - 600) ** 2 + (MT_y - 200) ** 2) ** 0.5
            d3=((MT_x - 600) ** 2 + (MT_y - 600) ** 2) ** 0.5
            BT_LOS_s=[[200,200,d0+random.gauss(0, 10)],[200,600,d1+random.gauss(0, 10)],[600,200,d2+random.gauss(0, 10)],[600,600,d3+random.gauss(0, 10)]]
            lls_s=LOS.Program(BT_LOS_s)
            pos1=lls_s.LLS_1_E()
            noises = ((pos1[0] - MT_x) ** 2 + (pos1[1] - MT_y) ** 2) ** 0.5
            squ.append(noises)
            '''
            lls=LOS.Program(BT_LOS_c)
            pos_lls=lls.LLS_1_E()
            noise=((pos_lls[0]-MT_x)**2+(pos_lls[1]-MT_y)**2)**0.5
            lls_noise.append(noise)

            #pos_aml = lls.estimate()
            #noise = ((pos_aml[0] - MT_x) ** 2 + (pos_aml[1] - MT_y) ** 2) ** 0.5
            #aml_noise.append(noise)

            #pos_srls = lls.estimate()
            #noise = ((pos_srls[0] - MT_x) ** 2 + (pos_srls[1] - MT_y) ** 2) ** 0.5
            #sr_ls_noise.append(noise)

        lls_Noise.append(sum(lls_noise)/10000)
        #Aml_noise.append(sum(aml_noise)/100)
        #SR_LS_noise.append(sum(sr_ls_noise) / 100)
        #print(sum(squ)/10000)
    name=['Inside','Outside']
    plt.bar(range(len(lls_Noise)), lls_Noise, tick_label=name)
    plt.xlabel('inside and Outside')
    plt.ylabel('Average_Bias/m')
    plt.title('Noise-Inside and Outside')
    plt.show()
    '''
    print(lls_Noise)
    print(lls_Noise)
    x=[i for i in range(200,1100,100)]
    plt.plot(x, lls_Noise,label='LLS')
    #plt.plot(x, Aml_noise, label='AML')
    #plt.plot(x, sr_ls_noise, label='SR-LS')
    plt.xlabel('Radius')
    plt.ylabel('Average_Bias/m')
    plt.title('Noise-Radius')
    plt.show()
    '''

def plot_cicle(centers,rads,MT,BT_x,BT_y):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    i=0
    for center,rad in zip(centers,rads):
        cir=plt.Circle((center[0],center[1]),radius=rad,color='y',fill=False)
        plt.plot([center[0], MT[0]], [center[1], MT[1]])
        plt.text((center[0]+MT[0])/2-5,(center[1]+MT[1])/2+8,'d'+str(i))
        ax.add_patch(cir)
        i=i+1
    plt.plot([centers[0][0] - rads[0], centers[0][0]], [centers[0][1], centers[0][1]])
    plt.text((centers[0][0] - rads[0] + centers[0][0]) / 2, (centers[0][1] + centers[0][1]) / 2 + 5, 'd0' + '\'')

    plt.plot([centers[1][0], centers[1][0]], [centers[1][1]-rads[1], centers[1][1]])
    plt.text((centers[1][0] + centers[1][0]) / 2+5, (centers[1][1]-rads[1] + centers[1][1]) / 2, 'd1' + '\'')

    plt.plot([centers[2][0], centers[2][0]], [centers[2][1] +rads[2], centers[2][1]])
    plt.text((centers[2][0] + centers[2][0]) / 2 + 5, (centers[2][1] +rads[2] + centers[1][1]) / 2, 'd2' + '\'')

    l3,=ax.plot(BT_x,BT_y,'bs')
    l1,=ax.plot(MT[0],MT[1],'ro')
    l2,=ax.plot(390,290,'k*')
    plt.legend(handles=[l1, l2,l3], labels=['true position', 'estimate','base station'], loc='best')
    plt.axis('equal')
    plt.axis('off')
    plt.show()

def main():
    para={}
    para['BT']=0
    para['BT_NLOS']=0
    para['MT_x'],para['MT_y']=0,0
    para['ave']=0
    para['sigma']=5
    para['lamda']=4
    para['BT_LOS']=[[251,251,0],[751,251,0],[251,751,0],[751,751,0]]
    #print('hello')
    ana = Analysis(para)
    ana.Heatmap()
    #ana.Change_Noise()
    #ana.losNlos()
if __name__=='__main__':
    main()