import matplotlib.pyplot as plt
import array
from mpl_toolkits.mplot3d import Axes3D

def withintraplot(filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs = array.array('f')
    ys = array.array('f')
    zs = array.array('f')
    ss = array.array('f')
    with open(filename, 'r') as datapoints:
        for line in datapoints.readlines():
            points = [eval(word) for word in line.split()]
            zs.append(points[0])
            ys.append(points[1])
            xs.append(points[2])
            ss.append(200 * points[5])
    ax.scatter(xs, ys, zs, s=ss)
    ax.set_xlabel('receiver variability')
    ax.set_ylabel('sender variability')
    ax.set_zlabel('common interest')
    plt.title("proportion of games in sample with at least one information-using equilibrium\n(larger size means larger proportion)")
    plt.show()

def withintraplotsender(filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs = array.array('f')
    ys = array.array('f')
    zs = array.array('f')
    ss = array.array('f')
    with open(filename, 'r') as datapoints:
        for line in datapoints.readlines():
            points = [eval(word) for word in line.split()]
            xs.append(points[0])
            ys.append(points[1])
            zs.append(points[4])
    ax.plot_wireframe(xs, ys, zs, rstride=0.33, cstride=0.5)
    ax.set_xlabel('common interest')
    ax.set_ylabel('sender variability')
    plt.title("proportion of games in sample with at least one information-using equilibrium")
    plt.show()

def withintraplotreceiver(filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs = array.array('f')
    ys = array.array('f')
    zs = array.array('f')
    ss = array.array('f')
    with open(filename, 'r') as datapoints:
        for line in datapoints.readlines():
            points = [eval(word) for word in line.split()]
            xs.append(points[0])
            ys.append(points[1])
            zs.append(points[4])
    ax.plot_wireframe(xs, ys, zs, rstride=0.33, cstride=0.5)
    ax.set_xlabel('common interest')
    ax.set_ylabel('receiver variability')
    plt.title("proportion of games in sample with at least one information-using equilibrium")
    plt.show()
   
   

def proportion(filename):
    ktd = array.array('f')
    proportion = array.array('f')
    with open(filename, 'r') as datapoints:
        for line in datapoints.readlines():
            points = [eval(word) for word in line.split()]
            ktd.append((3 - points[0])/3)
            proportion.append(points[1])
    plt.xlabel('Common interest')
    plt.ylabel('Proportion of games with informative equilibria')
    plt.plot(ktd, proportion)
    plt.show()

    
def champion(filename):
    ktd = array.array('f')
    champion = array.array('f')
    #champion95 = array.array('f')
    #champion50 = array.array('f')
    with open(filename, 'r') as datapoints:
        for line in datapoints.readlines():
            points = [eval(word) for word in line.split()]
            ktd.append((3 - points[0])/3)
            champion.append(points[1])
            #champion95.append(points[2])
            #champion50.append(points[3])
    plt.xlabel('Common interest')
    plt.ylabel('Maximum mutual information between states and acts')
    plt.plot(ktd, champion)
    plt.show()
