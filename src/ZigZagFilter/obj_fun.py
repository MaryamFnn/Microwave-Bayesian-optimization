
from webbrowser import UnixBrowser
import blackboxzig
import numpy as np
import matplotlib.pyplot as plt



def obj_fun(x=[None], debug=False):   ## x is the list of our parameters
    y = []
    gg=0
    ggg=0
    gggg=0
    for i, xi in enumerate(x):
        f ,s21 = blackboxzig.blackbox(xi, debug=debug, simm='linear',fmin = 1 , fmax = 4.5)

        plt.plot(f, 20*np.log10(np.abs(s21)))
        plt.xlabel('freq(Hz)')
        plt.ylabel('S21')
        plt.savefig(f".\\figures\\S21Responces.png")
        # with open ("log2.txt",'a') as O:
        #     O.write(f"f={f}\n----------------\n")
        #     O.write(f"s21={s21}\n----------------\n")

        farr=np.array(f)
        s21abs = np.abs(s21)

        BW = np.array([2.45e9,2.55e9])
        UBW = (farr < BW[0]).flatten()
        LBW = (farr > BW[1]).flatten()
        BW = np.bitwise_and((farr>=BW[0]),(farr<=BW[1])).flatten()
        y_BW = s21abs[BW]
        y_UBW = s21abs[UBW]
        y_LBW = s21abs[LBW]

        #y_BW_new = 10*(y_BW-.708)
        #y_UBW_new = .01-y_UBW
        #y_LBW_new = .01-y_LBW

        #y11=np.hstack((y_UBW_new,y_BW_new,y_LBW_new))
        #y1=np.sum(y11)

        y1=10*(np.sum(y_BW-.708))+np.sum(.01-y_LBW)+np.sum(.01-y_UBW)
        #ynew=np.concatenate(y_UBW,y_BW,y_LBW)


        #with open ("log2.txt",'a') as O:
         #     O.write(f"ynew={ynew}\n----------------\n")
        #      O.write(f"lbw={LBW}\n----------------\n")
        #      O.write(f"bw={BW}\n----------------\n")
        #      O.write(f"s21abs={s21abs}\n----------------\n")
        #      O.write(f"sbw={y_BW}\n----------------\n")
        #      O.write(f"subw={y_UBW}\n----------------\n")
        #      O.write(f"slbw={y_LBW}\n----------------\n")
        #      O.write(f"y1={y1}\n----------------\n")

        # # BW = np.array([2.45,2.55])
        # UBW = (f > BW[0]).flatten()
        # LBW =(f < BW[0]).flatten()
        # with open ("log2.txt",'a') as O:
        #     O.write(f"UBW={UBW}\n----------------\n")
        #     O.write(f"LBW={LBW}\n----------------\n")






        # for i , fi in enumerate(f):
        #     # with open ("log2.txt",'a') as O:
        #     #          O.write(f"f1={fi}\n----------------\n")
        #     if fi < 2.45e9  :
        #         with open ("log1.txt",'a') as O:
        #               O.write(f"f1={fi}\n----------------\n")
        #         gg = gg+(0.01-abs(s21[i]))
        #     if fi > 2.55e9  :
        #         with open ("log2.txt",'a') as O:
        #               O.write(f"f1={fi}\n----------------\n")
        #         ggg = ggg+(0.01-abs(s21[i]))
        #     if 2.45e9 <=fi <= 2.55e9 :
        #         with open ("log3.txt",'a') as O:
        #             O.write(f"f2={fi}\n----------------\n")
        #         gggg = gggg+10*(abs(s21[i])-0.708)

        # for k in range(0,a) or range(b+1,len(f)+1):
        #     gg = gg+(.01-abs(s21[k]))
        #     #g1.append(gg)
        # for j in range(a,b+1):
        #     ggg = ggg+10*(abs(s21[j])-0.708)
        #     #g2.append(ggg)


        #gtot=gg+ggg+gggg  # to maximize

        with open ("log.txt",'a') as O:

            O.write(f"y={y1}\n----------------\n")


        y.append(y1)

    return  np.vstack(y)

if __name__=='__main__':
    f = obj_fun(x=[[18,.9]],debug=False)