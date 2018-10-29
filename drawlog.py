import matplotlib.pyplot as plt

logfilepath = './cache/model/depth_gan/H36M_dummy/log.txt'

epoch = []
time = []
DisLoss = []
GenLoss = []
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
            DisLoss.append(0)
            GenLoss.append(0)
fig = plt.figure()
axis = plt.gca()
plt.ylim(0, 1)
plt.plot(epoch, DisLoss)
plt.plot(epoch, GenLoss)
plt.legend()
plt.show()
