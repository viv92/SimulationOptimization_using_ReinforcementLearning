import simpy
import pygame
import numpy as np
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#define all time delays

def EnterAreaA_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.uniform(0.3, 0.34))
def EnterAreaB_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.uniform(0.22, 0.26))
def DumpBucketA_time():
    return np.random.uniform(0.32, 0.34)
def DumpBucketB_time():
    return np.random.uniform(0.32, 0.34)
def ExcavateA_time():
    return np.random.uniform(0.33, 0.37)
def ExcavateB_time():
    return np.random.uniform(0.33, 0.37)
def Haul_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.triangular(4.4, 5.7, 6.6))
def EnterDump_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.uniform(0.6, 0.8))
def Dump_time():
    return np.random.triangular(2, 2.1, 2.2)
def Return0_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.triangular(2.5, 2.9, 3.4))
def Return1A_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.triangular(1, 1.5, 2.0))
def Return1B_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.triangular(1.8, 2.3, 2.8))

#define all processes

def DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity):
    while TrkUndrExcA.level == 0:
        yield env.timeout(1)
    #print "In DumpBucketA, TrkUndrExcA=%d" % TrkUndrExcA.level
    yield ExcWtDmpA.get(1)
    #print "starting DumpBucketA at %.2f" % env.now
    yield env.timeout(DumpBucketA_time())
    yield SlInTrkA.put(BucketA_capacity)
    env.process(ExcavateA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity))
    #print "finished DumpBucketA with SlInTrkA=%.2f at %.2f" % (SlInTrkA.level, env.now)

def DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity):
    while TrkUndrExcB.level == 0:
        yield env.timeout(1)
    #print "In DumpBucketB, TrkUndrExcB=%d" % TrkUndrExcB.level
    yield ExcWtDmpB.get(1)
    #print "starting DumpBucketB at %.2f" % env.now
    yield env.timeout(DumpBucketB_time())
    yield SlInTrkB.put(BucketB_capacity)
    env.process(ExcavateB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity))
    #print "finished DumpBucketB with SlInTrkB=%.2f at %.2f" % (SlInTrkB.level, env.now)

def ExcavateA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity):
    #print "starting ExcavateA at %.2f" % env.now
    yield env.timeout(ExcavateA_time())
    #print "finished ExcavateA at %.2f" % env.now
    yield ExcWtDmpA.put(1)
    env.process(DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity))
    #print "finished ExcavateA at %.2f" % env.now

def ExcavateB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity):
    #print "starting ExcavateB at %.2f" % env.now
    yield env.timeout(ExcavateB_time())
    #print "finished ExcavateB at %.2f" % env.now
    yield ExcWtDmpB.put(1)
    env.process(DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity))

def EnterAreaA(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy):
    global soil_ready_to_dump

    yield TrkUndrExcA.put(1) #will move forward only if TrkUndrExc.level was 0, also blocks other trucks
    #print "%s inside EnterAreaA - put 1 TrkUndrExcA. TrkUndrExcA.level=%d at %.2f" % (name, TrkUndrExcA.level, env.now)
    yield ManeuvSpcA.get(1)
    #print "%s inside EnterAreaA - got 1 ManeuvSpcA. ManeuvSpcA.level=%d at %.2f" % (name, ManeuvSpcA.level, env.now)
    yield TrkWtLdA.get(1)
    #print "%s starting EnterAreaA at %.2f" % (name, env.now)
    yield env.timeout(EnterAreaA_time(TruckSpdRatio))
    #print "-----%s finished EnterAreaA at %.2f" % (name, env.now)
    soil_ready_to_dump += TruckCap
    yield ManeuvSpcA.put(1)
    #print "%s post EnterAreaA - put 1 ManeuvSpcA. ManeuvSpcA.level=%d at %.2f" % (name, ManeuvSpcA.level, env.now)
    env.process(HaulA(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))

def EnterAreaB(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy):
    global soil_ready_to_dump

    #while (DmpdSoil.level + soil_ready_to_dump) > (SoilAmt - TruckCap): #wait here if SoilAmt met
    #    yield env.timeout(1)
    yield TrkUndrExcB.put(1) #will move forward only if TrkUndrExc.level was 0, also blocks other trucks
    #print "%s inside EnterAreaB - put 1 TrkUndrExcB. TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcB.level, env.now)
    yield ManeuvSpcB.get(1)
    #print "%s inside EnterAreaB - got 1 ManeuvSpcB. ManeuvSpcB.level=%d at %.2f" % (name, ManeuvSpcB.level, env.now)
    yield TrkWtLdB.get(1)
    #print "%s starting EnterAreaB at %.2f" % (name, env.now)
    yield env.timeout(EnterAreaB_time(TruckSpdRatio))
    #print "-----%s finished EnterAreaB at %.2f" % (name, env.now)
    soil_ready_to_dump += TruckCap
    yield ManeuvSpcB.put(1)
    #print "%s post EnterAreaB - put 1 ManeuvSpcB. ManeuvSpcB.level=%d at %.2f" % (name, ManeuvSpcB.level, env.now)
    env.process(HaulB(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))

def HaulA(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy):
    global num_of_load

    yield SlInTrkA.get(TruckCap)
    yield TrkUndrExcA.get(1)
    #print "%s inside Haul - got 1 TrkUndrExc. TrkUndrExcA.level=%d; TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcA.level, TrkUndrExcB.level, env.now)
    #print "-----%s starting Haul at %.2f" % (name, env.now)
    num_of_load += 1

    yield env.timeout(Haul_time(TruckSpdRatio))
    #print "%s finished Haul at %.2f" % (name, env.now)
    yield WtEnterDump.put(1)
    #print "%s post Haul - put 1 WtEnterDump. WtEnterDump.level=%d at %.2f" % (name, WtEnterDump.level, env.now)
    env.process(EnterDump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))

def HaulB(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy):
    global num_of_load

    yield SlInTrkB.get(TruckCap)
    yield TrkUndrExcB.get(1)
    #print "%s inside Haul - got 1 TrkUndrExc. TrkUndrExcA.level=%d; TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcA.level, TrkUndrExcB.level, env.now)
    #print "-----%s starting Haul at %.2f" % (name, env.now)
    num_of_load += 1

    yield env.timeout(Haul_time(TruckSpdRatio))
    #print "%s finished Haul at %.2f" % (name, env.now)
    yield WtEnterDump.put(1)
    #print "%s post Haul - put 1 WtEnterDump. WtEnterDump.level=%d at %.2f" % (name, WtEnterDump.level, env.now)
    env.process(EnterDump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))


def EnterDump(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy):
    yield DumpSpots.get(1)
    #print "%s Inside EnterDump - got 1 DumpSpots. DumpSpots.level=%d at %.2f" % (name, DumpSpots.level, env.now)
    yield WtEnterDump.get(1)
    #print "%s Inside EnterDump - got 1 WtEnterDump. WtEnterDump.level=%d at %.2f" % (name, WtEnterDump.level, env.now)
    #print "%s starting EnterDump at %.2f" % (name, env.now)
    yield env.timeout(EnterDump_time(TruckSpdRatio))
    #print "%s finished EnterDump at %.2f" % (name, env.now)
    env.process(Dump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))


def Dump(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy):
    global num_of_dump
    global soil_ready_to_dump
    #print "%s starting Dump at %.2f" % (name, env.now)
    yield env.timeout(Dump_time())
    #print "%s finished Dump at %.2f" % (name, env.now)
    yield DmpdSoil.put(TruckCap)
    soil_ready_to_dump -= TruckCap
    #print "%s post Dump - put %d DmpdSoil. DmpdSoil.level=%d at %.2f" % (name, TruckCap, DmpdSoil.level, env.now)
    yield DumpSpots.put(1)
    #print "%s post Dump - put 1 DumpSpots. DumpSpots.level=%d at %.2f" % (name, DumpSpots.level, env.now)
    env.process(Return0(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))
    num_of_dump += 1


def Return0(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy):
    global num_of_return
    #print "%s starting Return0 at %.2f" % (name, env.now)
    yield env.timeout(Return0_time(TruckSpdRatio))
    #print "%s finished Return0 at %.2f" % (name, env.now)
    action = policy()
    if action:
        env.process(Return1A(env, name, TrkWtLdA, TrkWtLdB,
        ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
        ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))
    else:
        env.process(Return1B(env, name, TrkWtLdA, TrkWtLdB,
        ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
        ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))
    num_of_return += 1


def Return1A(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy):
    #print "%s starting Return1A at %.2f" % (name, env.now)
    yield env.timeout(Return1A_time(TruckSpdRatio))
    #print "-----%s finished Return1A at %.2f" % (name, env.now)
    yield TrkWtLdA.put(1)
    #print "%s post Return1A - put 1 TrkWtLdA. TrkWtLdA.level=%d at %.2f" % (name, TrkWtLdA.level, env.now)
    env.process(EnterAreaA(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))

def Return1B(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy):
    #print "%s starting Return1B at %.2f" % (name, env.now)
    yield env.timeout(Return1B_time(TruckSpdRatio))
    #print "-----%s finished Return1B at %.2f" % (name, env.now)
    yield TrkWtLdB.put(1)
    #print "%s post Return1B - put 1 TrkWtLdB. TrkWtLdB.level=%d at %.2f" % (name, TrkWtLdB.level, env.now)
    env.process(EnterAreaB(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))

#define monitoring process
def monitor(env, DmpdSoil, TrkWtLdA, TrkWtLdB, SoilAmt, nTrucks, TruckCap):
    global num_of_load
    global num_of_dump
    global num_of_return

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
        #print "\nobs_num=%d\t DmpdSoil=%d\t TrkWtLdA=%d\t TrkWtLdB=%d\n" % (
        #obs_num, DmpdSoil.level, TrkWtLdA.level, TrkWtLdB.level)

        #terminate condition
        if (DmpdSoil.level > (SoilAmt-TruckCap)):
            print "\nDmpdSoil.level = %d. Desired SoilAmt achieved" % DmpdSoil.level

            #calculate outputs
            L_hrs = env.now/60.0
            L_hourlyCst = OHCst+ExcCst+(TrckCst*nTrucks) #duh constant
            L_prodRate = DmpdSoil.level/L_hrs
            L_unitCst = L_hourlyCst/L_prodRate
            Hrs.append(L_hrs)
            HourlyCst.append(L_hourlyCst)
            ProdRate.append(L_prodRate)
            UnitCst.append(L_unitCst)

            print """
            nTrucks = %d\n
            TruckCap = %d\n
            num_of_load = %d \n
            num_of_dump = %d \n
            num_of_return = %d \n
            soil_ready_to_dump = %d\n
            Hrs = %.4f \n
            HourlyCst = %.4f \n
            ProdRate = %.4f \n
            UnitCst = %.4f
            """ % (nTrucks, TruckCap, num_of_load, num_of_dump, num_of_return, soil_ready_to_dump,
            L_hrs, L_hourlyCst, L_prodRate, L_unitCst)
            return

        yield env.timeout(1)


#seed
np.random.seed(0)

#global variables
num_of_load = 0
num_of_dump = 0
num_of_return = 0
soil_ready_to_dump = 0

#input global params
SoilAmt = 10000
TrckCst = 48
ExcCst = 65
OHCst = 75

#output global params (results / costs)
HourlyCst = []
Hrs = []
ProdRate = []
UnitCst = []

def run_sim(nTrucks, BucketA_capacity, BucketB_capacity, Truck1_capacity, Truck2_capacity, Truck1_speedRatio, Truck2_speedRatio, policy):

    #simulation environment
    env = simpy.Environment()
    #resources
    TrkWtLdA = simpy.Container(env, init=(nTrucks/2), capacity=nTrucks)
    TrkWtLdB = simpy.Container(env, init=nTrucks-(nTrucks/2), capacity=nTrucks)
    ManeuvSpcA = simpy.Container(env, init=1, capacity=1)
    ManeuvSpcB = simpy.Container(env, init=1, capacity=1)
    TrkUndrExcA = simpy.Container(env, init=0, capacity=1)
    TrkUndrExcB = simpy.Container(env, init=0, capacity=1)
    SlInTrkA = simpy.Container(env, init=0, capacity=(max(Truck1_capacity,Truck2_capacity) + BucketA_capacity))
    SlInTrkB = simpy.Container(env, init=0, capacity=(max(Truck1_capacity,Truck2_capacity) + BucketB_capacity))
    ExcWtDmpA = simpy.Container(env, init=1, capacity=1)
    ExcWtDmpB = simpy.Container(env, init=1, capacity=1)
    WtEnterDump = simpy.Container(env, init=0, capacity=nTrucks)
    DumpSpots = simpy.Container(env, init=3, capacity=3)
    DmpdSoil = simpy.Container(env, init=0, capacity=SoilAmt)
    #dump bucket processes
    env.process(DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity))
    env.process(DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity))

    #initial truck arrangement - half of type1 and half of type2
    for i in range(nTrucks):
        if i < (nTrucks/2):
            TruckCap = Truck1_capacity
            TruckSpdRatio = Truck1_speedRatio
        else:
            TruckCap = Truck2_capacity
            TruckSpdRatio = Truck2_speedRatio
        if i % 2 == 0:
            #truck moving process for each truck
            env.process(EnterAreaA(env, 'truck %d' % i, TrkWtLdA, TrkWtLdB,
            ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
            ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))
        else:
            #truck moving process for each truck
            env.process(EnterAreaB(env, 'truck %d' % i, TrkWtLdA, TrkWtLdB,
            ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
            ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, policy))

    #monitoring process - responsible for terminating the simulation / episode
    proc = env.process(monitor(env, DmpdSoil, TrkWtLdA, TrkWtLdB, SoilAmt, nTrucks, max(Truck1_capacity, Truck2_capacity)))
    #run all processes
    env.run(until=proc)


#main
def main():

    nTrucks = 10
    BucketA_capacity = 1.5
    BucketB_capacity = 1.0
    Truck1_capacity = 6
    Truck2_capacity = 3
    Truck1_speed = 15.0
    Truck2_speed = 20.0
    Truck1_speedRatio = Truck1_speed / (Truck1_speed + Truck2_speed)
    Truck2_speedRatio = Truck2_speed / (Truck1_speed + Truck2_speed)

    #run simulation
    run_sim(nTrucks, BucketA_capacity, BucketB_capacity, Truck1_capacity, Truck2_capacity, Truck1_speedRatio, Truck2_speedRatio)

if __name__ == '__main__':
    main()
