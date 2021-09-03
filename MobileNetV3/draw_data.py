from matplotlib import pyplot as plt
def smooth_line(li,size):
    retract = (size-1)//2
    for i in range(retract,len(li)-retract):
        li[i] = sum(li[i-retract:i+retract])/size
    return li
def draw_loss(losslist,epoch,savepath):
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    temp_color_x = [x+1 for x in range(len(losslist))]
    losslist = smooth_line(losslist,100)
    plt.plot(temp_color_x,losslist)
    
    plt.xticks(range(0,len(losslist),5*len(losslist)//epoch),range(0,epoch,5))
    plt.savefig(savepath)
    print("已经保存损失变化图像%s"%(savepath))
    pass





