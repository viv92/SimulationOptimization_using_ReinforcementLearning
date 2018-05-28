import simpy
import numpy as np
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#define all time delays

def EnterArea_time():
    return 0.15
def DumpBucket_time():
    return np.random.uniform(0.06, 0.1)
def SwingEmpty_time():
    return 0.14
def Excavate_time():
    return np.random.uniform(0.08, 0.12)
def SwingLoaded_time():
    return 0.15
def Haul1_time():
    return np.random.triangular(2.2, 2.85, 3.3)
def EnterLd_time():
    return 0.3
def FinishLd_time():
    return 1.45
def Haul2_time():
    return np.random.triangular(5.25, 7, 8.25)
def Dump_time():
    return np.random.triangular(1, 1.05, 1.1)
def Return1_time():
    return np.random.triangular(1.25, 1.45, 1.7)
def EnterEt_time():
    return 0.3
def FinishEt_time():
    return 1.45
def Return2_time():
    return np.random.triangular(3.75, 4.5, 5.25)

#define all processes

def DumpBucket(env, TrkUndrExc, ExcWtDmp, SlInTrk):
    while TrkUndrExc.level == 0:
        yield env.timeout(1)
    #print "In DumpBucket, TrkUndrExc=%d" % TrkUndrExc.level
    yield ExcWtDmp.get(1)
    #print "starting DumpBucket at %.2f" % env.now
    yield env.timeout(DumpBucket_time())
    yield SlInTrk.put(2.5)
    env.process(SwingEmpty(env, TrkUndrExc, ExcWtDmp, SlInTrk))
    #print "finished DumpBucket with SlInTrk=%.2f at %.2f" % (SlInTrk.level, env.now)

def SwingEmpty(env, TrkUndrExc, ExcWtDmp, SlInTrk):
    #print "starting SwingEmpty at %.2f" % env.now
    yield env.timeout(SwingEmpty_time())
    env.process(Excavate(env, TrkUndrExc, ExcWtDmp, SlInTrk))
    #print "finished SwingEmpty at %.2f" % env.now

def Excavate(env, TrkUndrExc, ExcWtDmp, SlInTrk):
    #print "starting Excavate at %.2f" % env.now
    yield env.timeout(Excavate_time())
    env.process(SwingLoaded(env, TrkUndrExc, ExcWtDmp, SlInTrk))
    #print "finished Excavate at %.2f" % env.now

def SwingLoaded(env, TrkUndrExc, ExcWtDmp, SlInTrk):
    #print "starting SwingLoaded at %.2f" % env.now
    yield env.timeout(SwingLoaded_time())
    yield ExcWtDmp.put(1)
    env.process(DumpBucket(env, TrkUndrExc, ExcWtDmp, SlInTrk))
    #print "finished SwingLoaded at %.2f" % env.now

def EnterArea(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    global soil_ready_to_dump
    while (DmpdSoil.level + soil_ready_to_dump) > (SoilAmt - truckCap): #wait here if SoilAmt met
        yield env.timeout(1)
    yield TrkUndrExc.put(1) #will move forward only if TrkUndrExc.level was 0, also blocks other trucks
    #print "%s inside EnterArea - put 1 TrkUndrExc. TrkUndrExc.level=%d at %.2f" % (name, TrkUndrExc.level, env.now)
    yield ManeuvSpc.get(1)
    #print "%s inside EnterArea - got 1 ManeuvSpc. ManeuvSpc.level=%d at %.2f" % (name, ManeuvSpc.level, env.now)
    yield TrkWtLd.get(1)
    #print "%s starting EnterArea at %.2f" % (name, env.now)
    yield env.timeout(EnterArea_time())
    #print "%s finished EnterArea at %.2f" % (name, env.now)
    soil_ready_to_dump += truckCap
    yield ManeuvSpc.put(1)
    #print "%s post EnterArea - put 1 ManeuvSpc. ManeuvSpc.level=%d at %.2f" % (name, ManeuvSpc.level, env.now)
    env.process(Haul1(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))


def Haul1(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    global num_of_load
    yield SlInTrk.get(truckCap)
    yield TrkUndrExc.get(1)
    #print "%s inside Haul1 - got 1 TrkUndrExc. TrkUndrExc.level=%d at %.2f" % (name, TrkUndrExc.level, env.now)
    #print "%s starting Haul1 at %.2f" % (name, env.now)
    yield env.timeout(Haul1_time())
    #print "%s finished Haul1 at %.2f" % (name, env.now)
    yield WtEnterLd.put(1)
    #print "%s post Haul1 - put 1 WtEnterLd. WtEnterLd.level=%d at %.2f" % (name, WtEnterLd.level, env.now)
    env.process(EnterLd(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))


def EnterLd(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    while EtSpots.level < 100:
        yield env.timeout(1)
    #print "%s Inside EnterLd - checked EtSpots.level=%d at %.2f" % (name, EtSpots.level, env.now)
    yield LdSpots.get(1)
    #print "%s Inside EnterLd - got 1 LdSpots. LdSpots.level=%d at %.2f" % (name, LdSpots.level, env.now)
    yield WtEnterLd.get(1)
    #print "%s Inside EnterLd - got 1 WtEnterLd. WtEnterLd.level=%d at %.2f" % (name, WtEnterLd.level, env.now)
    yield EntryPass.get(1)
    #print "%s Inside EnterLd - got 1 EntryPass. EntryPass.level=%d at %.2f" % (name, EntryPass.level, env.now)
    #print "%s starting EnterLd at %.2f" % (name, env.now)
    yield env.timeout(EnterLd_time())
    #print "%s finished EnterLd at %.2f" % (name, env.now)
    yield EntryPass.put(1)
    #print "%s post EnterLd - put 1 EntryPass. EntryPass.level=%d at %.2f" % (name, EntryPass.level, env.now)
    env.process(FinishLd(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))


def FinishLd(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    #print "%s starting FinishLd at %.2f" % (name, env.now)
    yield env.timeout(FinishLd_time())
    #print "%s finished FinishLd at %.2f" % (name, env.now)
    yield LdSpots.put(1)
    #print "%s post FinishLd - put 1 LdSpots. LdSpots.level=%d at %.2f" % (name, LdSpots.level, env.now)
    env.process(Haul2(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))


def Haul2(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    #print "%s starting Haul2 at %.2f" % (name, env.now)
    yield env.timeout(Haul2_time())
    #print "%s finished Haul2 at %.2f" % (name, env.now)
    env.process(Dump(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))


def Dump(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    global num_of_dump
    global soil_ready_to_dump
    #print "%s starting Dump at %.2f" % (name, env.now)
    yield env.timeout(Dump_time())
    #print "%s finished Dump at %.2f" % (name, env.now)
    yield DmpdSoil.put(truckCap)
    soil_ready_to_dump -= truckCap
    #print "%s post Dump - put 15 DmpdSoil. DmpdSoil.level=%d at %.2f" % (name, DmpdSoil.level, env.now)
    env.process(Return1(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))


def Return1(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    #print "%s starting Return1 at %.2f" % (name, env.now)
    yield env.timeout(Return1_time())
    #print "%s finished Return1 at %.2f" % (name, env.now)
    yield WtEnterEt.put(1)
    #print "%s post Return1 - put 1 WtEnterEt. WtEnterEt.level=%d at %.2f" % (name, WtEnterEt.level, env.now)
    env.process(EnterEt(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))


def EnterEt(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    while LdSpots.level < 100:
        yield env.timeout(1)
    #print "%s Inside EnterEt - checked LdSpots.level=%d at %.2f" % (name, LdSpots.level, env.now)
    yield EtSpots.get(1)
    #print "%s Inside EnterEt - got 1 EtSpots. EtSpots.level=%d at %.2f" % (name, EtSpots.level, env.now)
    yield WtEnterEt.get(1)
    #print "%s Inside EnterEt - got 1 WtEnterEt. WtEnterEt.level=%d at %.2f" % (name, WtEnterEt.level, env.now)
    yield EntryPass.get(1)
    #print "%s Inside EnterEt - got 1 EntryPass. EntryPass.level=%d at %.2f" % (name, EntryPass.level, env.now)
    #print "%s starting EnterEt at %.2f" % (name, env.now)
    yield env.timeout(EnterEt_time())
    #print "%s finished EnterEt at %.2f" % (name, env.now)
    yield EntryPass.put(1)
    #print "%s post EnterEt - put 1 EntryPass. EntryPass.level=%d at %.2f" % (name, EntryPass.level, env.now)
    env.process(FinishEt(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))


def FinishEt(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    #print "%s starting FinishEt at %.2f" % (name, env.now)
    yield env.timeout(FinishEt_time())
    #print "%s finished FinishEt at %.2f" % (name, env.now)
    yield EtSpots.put(1)
    #print "%s post FinishEt - put 1 EtSpots. EtSpots.level=%d at %.2f" % (name, EtSpots.level, env.now)
    env.process(Return2(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))


def Return2(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap):
    global num_of_return
    #print "%s starting Return2 at %.2f" % (name, env.now)
    yield env.timeout(Return2_time())
    #print "%s finished Return2 at %.2f" % (name, env.now)
    yield TrkWtLd.put(1)
    #print "%s post Return2 - put 1 TrkWtLd. TrkWtLd.level=%d at %.2f" % (name, TrkWtLd.level, env.now)
    env.process(EnterArea(env, name, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
    WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))

#define monitoring process
def monitor(env, DmpdSoil, TrkWtLd, SoilAmt, nTrucks, truckCap):
    #global cost params to be updated
    global TrckCst
    global ExcCst
    global OHCst
    global HourlyCst
    global Hrs
    global ProdRate
    global UnitCst
    #num of monitoring observations
    obs_num = 0
    #run monitoring loop
    while True:
        obs_num += 1
        #print "\nobs_num=%d\t DmpdSoil=%d\t TrkWtLd=%d\n" % (
        #obs_num, DmpdSoil.level, TrkWtLd.level)

        #terminate condition
        if (DmpdSoil.level > (SoilAmt-truckCap)):
            print "\nDmpdSoil.level = %d. Desired SoilAmt achieved" % DmpdSoil.level

            #calculate outputs
            L_hrs = env.now/60.0
            L_hourlyCst = OHCst+ExcCst+(TrckCst*nTrucks) #duh constant
            L_prodRate = DmpdSoil.level/L_hrs
            L_unitCst = L_hourlyCst/L_prodRate
            Hrs[nTrucks][truckCap] = L_hrs
            HourlyCst[nTrucks][truckCap] = L_hourlyCst
            ProdRate[nTrucks][truckCap] = L_prodRate
            UnitCst[nTrucks][truckCap] = L_unitCst

            print """
            nTrucks = %d\n
            truckCap = %d\n
            Hrs = %.4f \n
            HourlyCst = %.4f \n
            ProdRate = %.4f \n
            UnitCst = %.4f
            """ % (nTrucks, truckCap, L_hrs, L_hourlyCst, L_prodRate, L_unitCst)
            return

        yield env.timeout(1)


#seed
np.random.seed(0)

#global variables
soil_ready_to_dump = 0

#hyperparams (max limits)
max_Trucks = 20
max_Capacity = 20

#input global params
SoilAmt = 10000
TrckCst = 48
ExcCst = 65
OHCst = 75

#output global params (results / costs)
HourlyCst = np.zeros((max_Trucks+1,max_Capacity+1))
Hrs = np.zeros((max_Trucks+1,max_Capacity+1))
ProdRate = np.zeros((max_Trucks+1,max_Capacity+1))
UnitCst = np.zeros((max_Trucks+1,max_Capacity+1))

#main
def main():
    global soil_ready_to_dump
    nTrucks_array = np.arange(max_Trucks+1)
    truckCapacities_array = np.arange(max_Capacity+1)
    for truckCapacity in truckCapacities_array:
        if truckCapacity == 0:
            continue
        for nTrucks in nTrucks_array:
            if nTrucks == 0:
                continue
            soil_ready_to_dump = 0

            #run simulation
            run_sim(truckCapacity, nTrucks)

    #get minimum unit cost and optimal values at which its achieved
    c = np.copy(UnitCst)
    c[0][:] = 100
    c[:,0] = (np.ones(max_Trucks+1)*100)
    minUnitCst = np.min(c)
    minIndex = np.argmin(c)
    optTrucks = minIndex / (max_Trucks+1)
    optCap = minIndex % (max_Trucks+1)
    print "Minimum unit cost = %.4f at optTrucks = %d and optCapacity = %d" % (
    minUnitCst, optTrucks, optCap)

    #plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(truckCapacities_array, nTrucks_array)
    Z = UnitCst
    surf = ax.plot_surface(X, Y, Z, linewidth=0, rstride=1, cstride=1, cmap=cm.coolwarm, vmin=0.0, vmax=4.0)
    ax.set_ylabel('Num of Trucks')
    ax.set_xlabel('Truck Capacity')
    ax.set_zlabel('Unit Cost')
    ax.set_zlim(0,4)
    ax.set_title("Narrow Pass Simulation (Martinez Paper - Fig4)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def run_sim(truckCap, nTrucks):
    #simulation environment
    env = simpy.Environment()
    #resources
    TrkWtLd = simpy.Container(env, init=nTrucks, capacity=nTrucks)
    ManeuvSpc = simpy.Container(env, init=1, capacity=1)
    TrkUndrExc = simpy.Container(env, init=0, capacity=1)
    SlInTrk = simpy.Container(env, init=0, capacity=truckCap+3)
    ExcWtDmp = simpy.Container(env, init=1, capacity=1)
    WtEnterLd = simpy.Container(env, init=0, capacity=nTrucks)
    EtSpots = simpy.Container(env, init=100, capacity=100)
    EntryPass = simpy.Container(env, init=1, capacity=1)
    LdSpots = simpy.Container(env, init=100, capacity=100)
    WtEnterEt = simpy.Container(env, init=0, capacity=nTrucks)
    DmpdSoil = simpy.Container(env, init=0, capacity=SoilAmt)
    #dump bucket process
    env.process(DumpBucket(env, TrkUndrExc, ExcWtDmp, SlInTrk))

    for i in range(nTrucks):
        #truck moving process for each truck
        env.process(EnterArea(env, 'truck %d' % i, TrkWtLd, ManeuvSpc, TrkUndrExc, SlInTrk, ExcWtDmp,
        WtEnterLd, EtSpots, EntryPass, LdSpots, WtEnterEt, DmpdSoil, truckCap))

    #monitoring process
    proc = env.process(monitor(env, DmpdSoil, TrkWtLd, SoilAmt, nTrucks, truckCap))

    #run all processes
    env.run(until=proc)

if __name__ == '__main__':
    main()
