import matplotlib.pyplot as plt
import matplotlib.animation as anim

import os


def drawlog(keyword, id):
    logfilepath = './cache/model/depth_gan/H36M_lr='+keyword+'/log.txt'

    if os.path.exists(logfilepath) == False:
        return
    epoch = []
    time = []
    DisLoss = []
    GenLoss = []

    fig = plt.figure(id)
    ax = fig.add_subplot(111)

    with open(logfilepath, 'r') as f:
        next(f)
        for line in f:
            f, l = 0, 0
            f = line.find('=')+1
            l = line.find(',')
            epoch.append(eval(line[f:l]))
            f = line.find('=', l)+1
            l = line.find(' s', f)
            time.append(eval(line[f:l]))
            f = line.find('[', l)+1
            l = line.find(' ', f)-1
            if(f > 0):
                DisLoss.append(eval(line[f:l]))
                f = l+1
                l = line.find(']', l)
                GenLoss.append(eval(line[f:l]))
            else:
                DisLoss.append(DisLoss[-1])
                GenLoss.append(GenLoss[-1])

    ax.clear()
    ax.set_ylim(0, 1)
    ax.set_title(keyword)
    ax.plot(epoch, DisLoss)
    ax.plot(epoch, GenLoss)
    plt.show()


# keys = ['0.01', '0.001', '0.03', '0.003', '0.005', '0.0005', '0.008', '0.0008']
# for i, key in enumerate(keys):
#     drawlog(key, i)

drawlog("0.03", 1)