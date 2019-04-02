import simpy
import numpy as np
import tensorflow as tf
import scipy
from collections import defaultdict
import sys
import os
import itertools
import matplotlib
matplotlib.use('Agg')

from lib import plotting
matplotlib.style.use('ggplot')

#define all time delays

def EnterAreaA_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.uniform(0.3, 0.34))
def EnterAreaB_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.uniform(0.3, 0.34))
def DumpBucketA_time():
    return np.random.uniform(0.16, 0.17)
def DumpBucketB_time():
    return np.random.uniform(0.32, 0.34)
def ExcavateA_time():
    return np.random.uniform(0.16, 0.17)
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
    global state
    global g_Truck1_capacity
    global g_Truck2_capacity

    while TrkUndrExcA.level == 0:
        yield env.timeout(1)
    #print "In DumpBucketA, TrkUndrExcA=%d" % TrkUndrExcA.level
    yield ExcWtDmpA.get(1)
    #print "starting DumpBucketA at %.2f" % env.now
    yield env.timeout(DumpBucketA_time())
    yield SlInTrkA.put(BucketA_capacity)
    if state[8] > 0:
        state[8] -= (BucketA_capacity/float(g_Truck1_capacity))
    else:
        state[9] -= (BucketA_capacity/g_Truck2_capacity)
    env.process(ExcavateA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity))
    #print "finished DumpBucketA with SlInTrkA=%.2f at %.2f" % (SlInTrkA.level, env.now)

def DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity):
    global state
    global g_Truck1_capacity
    global g_Truck2_capacity

    while TrkUndrExcB.level == 0:
        yield env.timeout(1)
    #print "In DumpBucketB, TrkUndrExcB=%d" % TrkUndrExcB.level
    yield ExcWtDmpB.get(1)
    #print "starting DumpBucketB at %.2f" % env.now
    yield env.timeout(DumpBucketB_time())
    yield SlInTrkB.put(BucketB_capacity)
    if state[10] > 0:
        state[10] -= (BucketB_capacity/float(g_Truck1_capacity))
    else:
        state[11] -= (BucketB_capacity/g_Truck2_capacity)
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
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    global g_Truck1_capacity

    #while (DmpdSoil.level + soil_ready_to_dump) > (SoilAmt - TruckCap): #wait here if SoilAmt met
    #    yield env.timeout(1)
    yield TrkUndrExcA.put(1) #will move forward only if TrkUndrExc.level was 0, also blocks other trucks
    #print "%s inside EnterAreaA - put 1 TrkUndrExcA. TrkUndrExcA.level=%d at %.2f" % (name, TrkUndrExcA.level, env.now)
    yield ManeuvSpcA.get(1)
    #print "%s inside EnterAreaA - got 1 ManeuvSpcA. ManeuvSpcA.level=%d at %.2f" % (name, ManeuvSpcA.level, env.now)
    yield TrkWtLdA.get(1)
    #print "%s starting EnterAreaA at %.2f" % (name, env.now)
    if TruckCap == g_Truck1_capacity:
        state[0] -= 1
        state[4] = 1
    else:
        state[1] -= 1
        state[5] = 1
    yield env.timeout(EnterAreaA_time(TruckSpdRatio))
    if TruckCap == g_Truck1_capacity:
        state[4] = 0
        state[8] = 1
    else:
        state[5] = 0
        state[9] = 1
    #print "-----%s finished EnterAreaA at %.2f" % (name, env.now)
    yield ManeuvSpcA.put(1)
    #print "%s post EnterAreaA - put 1 ManeuvSpcA. ManeuvSpcA.level=%d at %.2f" % (name, ManeuvSpcA.level, env.now)
    env.process(HaulA(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

def EnterAreaB(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    global g_Truck1_capacity

    #while (DmpdSoil.level + soil_ready_to_dump) > (SoilAmt - TruckCap): #wait here if SoilAmt met
    #    yield env.timeout(1)
    yield TrkUndrExcB.put(1) #will move forward only if TrkUndrExc.level was 0, also blocks other trucks
    #print "%s inside EnterAreaB - put 1 TrkUndrExcB. TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcB.level, env.now)
    yield ManeuvSpcB.get(1)
    #print "%s inside EnterAreaB - got 1 ManeuvSpcB. ManeuvSpcB.level=%d at %.2f" % (name, ManeuvSpcB.level, env.now)
    yield TrkWtLdB.get(1)
    #print "%s starting EnterAreaB at %.2f" % (name, env.now)
    if TruckCap == g_Truck1_capacity:
        state[2] -= 1
        state[6] = 1
    else:
        state[3] -= 1
        state[7] = 1
    yield env.timeout(EnterAreaB_time(TruckSpdRatio))
    if TruckCap == g_Truck1_capacity:
        state[6] = 0
        state[10] = 1
    else:
        state[7] = 0
        state[11] = 1
    #print "-----%s finished EnterAreaB at %.2f" % (name, env.now)
    yield ManeuvSpcB.put(1)
    #print "%s post EnterAreaB - put 1 ManeuvSpcB. ManeuvSpcB.level=%d at %.2f" % (name, ManeuvSpcB.level, env.now)
    env.process(HaulB(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

def HaulA(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    global num_of_load
    global g_Truck1_capacity

    yield SlInTrkA.get(TruckCap)
    yield TrkUndrExcA.get(1)
    #print "%s inside Haul - got 1 TrkUndrExc. TrkUndrExcA.level=%d; TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcA.level, TrkUndrExcB.level, env.now)
    #print "-----%s starting Haul at %.2f" % (name, env.now)
    num_of_load += 1
    if TruckCap == g_Truck1_capacity:
        state[8] = 0
    else:
        state[9] = 0
    yield env.timeout(Haul_time(TruckSpdRatio))
    #print "%s finished Haul at %.2f" % (name, env.now)
    yield WtEnterDump.put(1)
    #print "%s post Haul - put 1 WtEnterDump. WtEnterDump.level=%d at %.2f" % (name, WtEnterDump.level, env.now)
    env.process(EnterDump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

def HaulB(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    global num_of_load
    global g_Truck1_capacity

    yield SlInTrkB.get(TruckCap)
    yield TrkUndrExcB.get(1)
    #print "%s inside Haul - got 1 TrkUndrExc. TrkUndrExcA.level=%d; TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcA.level, TrkUndrExcB.level, env.now)
    #print "-----%s starting Haul at %.2f" % (name, env.now)
    num_of_load += 1
    if TruckCap == g_Truck1_capacity:
        state[10] = 0
    else:
        state[11] = 0
    yield env.timeout(Haul_time(TruckSpdRatio))
    #print "%s finished Haul at %.2f" % (name, env.now)
    yield WtEnterDump.put(1)
    #print "%s post Haul - put 1 WtEnterDump. WtEnterDump.level=%d at %.2f" % (name, WtEnterDump.level, env.now)
    env.process(EnterDump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))


def EnterDump(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    yield DumpSpots.get(1)
    #print "%s Inside EnterDump - got 1 DumpSpots. DumpSpots.level=%d at %.2f" % (name, DumpSpots.level, env.now)
    yield WtEnterDump.get(1)
    #print "%s Inside EnterDump - got 1 WtEnterDump. WtEnterDump.level=%d at %.2f" % (name, WtEnterDump.level, env.now)
    #print "%s starting EnterDump at %.2f" % (name, env.now)
    yield env.timeout(EnterDump_time(TruckSpdRatio))
    #print "%s finished EnterDump at %.2f" % (name, env.now)
    env.process(Dump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))


def Dump(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global num_of_dump
    #print "%s starting Dump at %.2f" % (name, env.now)
    yield env.timeout(Dump_time())
    #print "%s finished Dump at %.2f" % (name, env.now)
    yield DmpdSoil.put(TruckCap)
    #print "%s post Dump - put %d DmpdSoil. DmpdSoil.level=%d at %.2f" % (name, TruckCap, DmpdSoil.level, env.now)
    yield DumpSpots.put(1)
    #print "%s post Dump - put 1 DumpSpots. DumpSpots.level=%d at %.2f" % (name, DumpSpots.level, env.now)
    env.process(Return0(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))
    num_of_dump += 1


def Return0(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global num_of_return
    global big_wait_penalty
    global actions_performed_without_maintenance_A
    global actions_performed_without_maintenance_B
    global idle_count_A
    global idle_count_B
    global repair_downtime_remaining_A
    global repair_downtime_remaining_B
    global both_excavators_failed_flag

    #print "%s starting Return0 at %.2f" % (name, env.now)
    yield env.timeout(Return0_time(TruckSpdRatio))
    #print "%s finished Return0 at %.2f" % (name, env.now)
    action = agent(name, env.now)

    #check for case when both excavators failed
    if both_excavators_failed_flag == True:
        yield env.timeout(big_wait_penalty) #big wait penalty = 2 hrs
        #restore both excavators as fresh
        #print "############ BOTH EXCAVATORS REPAIRED  #####################"
        #print "---------1 remaining down times A:%d \t B:%d" % (repair_downtime_remaining_A, repair_downtime_remaining_B)
        both_excavators_failed_flag = False
        repair_downtime_remaining_A = 0
        repair_downtime_remaining_B = 0
        actions_performed_without_maintenance_A = 0
        actions_performed_without_maintenance_B = 0
        idle_count_A = 0
        idle_count_B = 0
        #print "---------2 remaining down times A:%d \t B:%d" % (repair_downtime_remaining_A, repair_downtime_remaining_B)

    if action == 0:
        env.process(Return1A(env, name, TrkWtLdA, TrkWtLdB,
        ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
        ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))
    else:
        env.process(Return1B(env, name, TrkWtLdA, TrkWtLdB,
        ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
        ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))
    num_of_return += 1


def Return1A(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    global g_Truck1_capacity

    #print "%s starting Return1A at %.2f" % (name, env.now)
    yield env.timeout(Return1A_time(TruckSpdRatio))
    #print "-----%s finished Return1A at %.2f" % (name, env.now)
    yield TrkWtLdA.put(1)
    if TruckCap == g_Truck1_capacity:
        state[0] += 1
    else:
        state[1] += 1
    #print "%s post Return1A - put 1 TrkWtLdA. TrkWtLdA.level=%d at %.2f" % (name, TrkWtLdA.level, env.now)
    env.process(EnterAreaA(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

def Return1B(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    global g_Truck1_capacity

    #print "%s starting Return1B at %.2f" % (name, env.now)
    yield env.timeout(Return1B_time(TruckSpdRatio))
    #print "-----%s finished Return1B at %.2f" % (name, env.now)
    yield TrkWtLdB.put(1)
    if TruckCap == g_Truck1_capacity:
        state[2] += 1
    else:
        state[3] += 1
    #print "%s post Return1B - put 1 TrkWtLdB. TrkWtLdB.level=%d at %.2f" % (name, TrkWtLdB.level, env.now)
    env.process(EnterAreaB(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

#define monitoring process
def monitor(env, DmpdSoil, TrkWtLdA, TrkWtLdB, SoilAmt, nTrucks, TruckCap):
    global num_of_load
    global num_of_dump
    global num_of_return
    global num_decisions
    global num_decisions_A
    global num_decisions_B

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

        #calculate outputs
        L_hrs = env.now/60.0
        L_hourlyCst = OHCst+ExcCst+(TrckCst*nTrucks) #duh constant
        L_prodRate = 0
        L_unitCst = 0
        if L_hrs > 0:
            L_prodRate = DmpdSoil.level/L_hrs
            #print "PRODRATE = ", L_prodRate
            if L_prodRate > 0:
                L_unitCst = L_hourlyCst/L_prodRate

        #update global stats
        Hrs.append(L_hrs)
        HourlyCst.append(L_hourlyCst)
        if L_prodRate < 100:
            ProdRate.append(L_prodRate)
        UnitCst.append(L_unitCst)

        #terminate condition
        if (DmpdSoil.level > (SoilAmt-TruckCap)):
            print "\nDmpdSoil.level = %d. Desired SoilAmt achieved" % DmpdSoil.level



            print """
            nTrucks = %d\n
            num_of_load = %d \n
            num_of_dump = %d \n
            num_of_return = %d \n
            Hrs = %.4f \n
            HourlyCst = %.4f \n
            ProdRate = %.4f \n
            UnitCst = %.4f \n
            num_decisions = %d \n
            num_decisions_A = %d \n
            num_decisions_B = %d \n
            ratio_A/B = %.4f
            """ % (nTrucks, num_of_load, num_of_dump, num_of_return,
            L_hrs, L_hourlyCst, L_prodRate, L_unitCst, num_decisions, num_decisions_A[-1], num_decisions_B[-1], (num_decisions_A[-1]/float(num_decisions_B[-1])))
            return


        yield env.timeout(1)


#seed
np.random.seed(0)

#global variables
num_of_load = 0
num_of_dump = 0
num_of_return = 0
nTrucks = 20
g_Truck1_capacity = 23
g_Truck2_capacity = 24

num_decisions = 0
num_decisions_A = []
num_decisions_B = []
local_decisions_A = []
local_decisions_B = []

idle_count_A = 0
idle_count_B = 0
actions_performed_without_maintenance_A = 0
actions_performed_without_maintenance_B = 0
repair_downtime_remaining_A = 0
repair_downtime_remaining_B = 0

repair_downtime = 45
maintenance_downtime = 6
actions_to_failure_without_maintenance = 30
both_excavators_failed_flag = False
big_wait_penalty = 60*60 #2 hrs penalty when both excavators fail


''' global state vector
state[0]: num of Trk1 WtLdA
state[1]: num of Trk2 WtLdA
state[2]: num of Trk1 WtLdB
state[3]: num of Trk2 WtLdB
state[4]: is Trk1 in ManeuvSpcA (0/1)
state[5]: is Trk2 in ManeuvSpcA (0/1)
state[6]: is Trk1 in ManeuvSpcB (0/1)
state[7]: is Trk2 in ManeuvSpcB (0/1)
state[8]: % empty Trk1 UndrExcA
state[9]: % empty Trk2 UndrExcA
state[10]: % empty Trk1 UndrExcB
state[11]: % empty Trk2 UndrExcB
state[12]: actions_performed_without_maintenance_A
state[13]: actions_performed_without_maintenance_B'''
state = np.zeros(12)
old_state = np.zeros((nTrucks,12))
old_time = np.zeros(nTrucks)
old_action = np.zeros(nTrucks).astype(int)
nA = 2 #number of actions
discount_factor = 0.99
alpha = 1e-3 #learning_rate
epsilon = 0.2 #epsilon for epsilon_greedy_policy

#get interactive session
sess = tf.InteractiveSession()
#placeholder for observation
obs = tf.placeholder(tf.float32, shape=[140])

#weights for action value function (in the form of a fully connected layer)
# fc = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(obs,0), num_outputs=140, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
# fc2 = tf.contrib.layers.fully_connected(inputs=fc, num_outputs=20, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
# output = tf.contrib.layers.fully_connected(inputs=fc2, num_outputs=nA, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

#weights for action value function (in the form of a LINEAR COMBINATION )
output = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(obs,0), num_outputs=nA, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

action_values = tf.squeeze(output)
#placeholder for chosen action
chosen_action = tf.placeholder(tf.int32, shape=[])
logit = tf.gather(action_values, chosen_action)
#placeholder for td target
target = tf.placeholder(tf.float32, shape=[])
#loss function for action value function
loss = tf.squared_difference(target, logit)
#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
#training op
train_op = optimizer.minimize(loss)
#global vars init
init = tf.global_variables_initializer()

#function to make epsilon greedy policy
def  make_epsilon_greedy_policy():
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        global nTrucks
        #convert observation to one-hot before feeding
        state_onehot = np.zeros(14*10)
        range_list = [nTrucks, nTrucks, nTrucks, nTrucks, 1, 1, 1, 1, 1, 1, 1, 1, actions_to_failure_without_maintenance, actions_to_failure_without_maintenance]
        for i in range(len(observation)):
            val = observation[i]
            index = int((val*9)/float(range_list[i]))
            state_onehot[(i*10) + index] = 1

        q_values = action_values.eval(feed_dict={obs: state_onehot})
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

# The policy we're following (its a functional)
policy_mu = make_epsilon_greedy_policy()

#Q-learning for now
def agent(truckName, time):
    global state #current state treated as next state
    global old_state #state for which we are learning
    global old_time
    global old_action
    global Mean_TD_Error
    global Iterations
    global num_decisions
    global num_decisions_A
    global num_decisions_B
    global local_decisions_A
    global local_decisions_B
    global idle_count_A
    global idle_count_B
    global actions_performed_without_maintenance_A
    global actions_performed_without_maintenance_B
    global repair_downtime_remaining_A
    global repair_downtime_remaining_B
    global repair_downtime
    global maintenance_downtime
    global actions_to_failure_without_maintenance
    global both_excavators_failed_flag
    global nTrucks

    truckIndex = int(truckName[len('truck')])
    reward = -1 * (time - old_time[truckIndex]) #time of cycle for this truck

    #modification to include maintenance in state vector
    state_vector_for_maintenance = np.append(state, np.append(actions_performed_without_maintenance_A, actions_performed_without_maintenance_B))
    old_state_vector_for_maintenance = np.append(old_state[truckIndex], np.append(actions_performed_without_maintenance_A, actions_performed_without_maintenance_B))

    #convert to one hot encoding
    state_onehot = np.zeros(14*10)
    old_state_onehot = np.zeros(14*10)
    range_list = [nTrucks, nTrucks, nTrucks, nTrucks, 1, 1, 1, 1, 1, 1, 1, 1, actions_to_failure_without_maintenance, actions_to_failure_without_maintenance]
    for i in range(len(state_vector_for_maintenance)):
        val = state_vector_for_maintenance[i]
        index = int((val*9)/float(range_list[i]))
        #print "I = ", i, "VAL = ", val, " INDEX = ", index, " FINAL = ", (i*10) + index
        # if val < 0:
        #     print "STATE: ", state
        # else:
        #     print "state: ", state
        state_onehot[(i*10) + index] = 1
    for i in range(len(old_state_vector_for_maintenance)):
        val = old_state_vector_for_maintenance[i]
        index = int((val*9)/float(range_list[i]))
        old_state_onehot[(i*10) + index] = 1


    if old_time[truckIndex] > 0:  #not the first ever decision - learn for the old_state
        cur_q_values = action_values.eval(feed_dict={obs: state_onehot})
        old_q_values = action_values.eval(feed_dict={obs: old_state_onehot})
        best_next_action = np.argmax(cur_q_values)
        td_target = reward + discount_factor * cur_q_values[best_next_action]
        td_error = td_target - old_q_values[old_action[truckIndex]]
        Iterations += 1
        if len(Mean_TD_Error) == 0:
            last_td_error = td_error
        else:
            last_td_error = abs(Mean_TD_Error[-1])
        Mean_TD_Error.append(((Iterations-1)*last_td_error + td_error)/Iterations)
        #update
        sess.run(train_op, feed_dict={obs: old_state_onehot, chosen_action: old_action[truckIndex], target: td_target})
        #get action for current state
        action_probs = policy_mu(state_vector_for_maintenance)
        action = np.random.choice(np.arange(nA), p=action_probs)
    else: #first ever decision - no learning since no old_action
        action = np.random.choice(np.arange(nA)) #take first action randomly
        Iterations = 1
    #set up for next decision call
    np.copyto(old_state_vector_for_maintenance, state_vector_for_maintenance)
    old_action[truckIndex] = action
    old_time[truckIndex] = time

    if num_decisions_A == [] and num_decisions_B == []:
        local_decisions_A = []
        local_decisions_B = []

    #print "---------- remaining down times A:%d \t B:%d" % (repair_downtime_remaining_A, repair_downtime_remaining_B)
    if repair_downtime_remaining_A > 0 and repair_downtime_remaining_B > 0: #both excavator under repair
        #print "############   BOTH EXCAVATORS FAILED  #####################"
        both_excavators_failed_flag = True #big time penalty = 2 hrs

    else:

        if repair_downtime_remaining_A > 0: #cant route to this excavavtor - under repair
            repair_downtime_remaining_A -= 1 #reduce remaining repair downtime
            #print "############ EXCAVATOR-A UNDER REPAIR: ACTIONS LEFT: %d #####################" % repair_downtime_remaining_A

            if action == 0:
                action = 1 #route to other excavator

            if repair_downtime_remaining_A == 0: # repair done - restore excavavtor as fresh
                #print "############ EXCAVATOR-A REPAIRED  #####################"
                actions_performed_without_maintenance_A = 0
                idle_count_A = 0

        if repair_downtime_remaining_B > 0: #cant route to this excavavtor - under repair
            repair_downtime_remaining_B -= 1 #reduce remaining repair downtime
            #print "############ EXCAVATOR-B UNDER REPAIR: ACTIONS LEFT: %d #####################" % repair_downtime_remaining_B

            if action == 1:
                action = 0 #route to other excavator

            if repair_downtime_remaining_B == 0: # repair done - restore excavavtor as fresh
                #print "############ EXCAVATOR-B REPAIRED  #####################"
                actions_performed_without_maintenance_B = 0
                idle_count_B = 0


        if action == 0:
            if len(num_decisions_A) == 0:
                num_decisions_A.append(1)
            else:
                num_decisions_A.append(num_decisions_A[-1] + 1)
            actions_performed_without_maintenance_A += 1
            idle_count_A = 0
            if repair_downtime_remaining_B == 0:
                idle_count_B += 1
        else:
            if len(num_decisions_B) == 0:
                num_decisions_B.append(1)
            else:
                num_decisions_B.append(num_decisions_B[-1] + 1)
            actions_performed_without_maintenance_B += 1
            idle_count_B = 0
            if repair_downtime_remaining_A == 0:
                idle_count_A += 1

        if len(num_decisions_A) == 0:
            local_decisions_A.append(0)
        else:
            local_decisions_A.append(num_decisions_A[-1])

        if len(num_decisions_B) == 0:
            local_decisions_B.append(0)
        else:
            local_decisions_B.append(num_decisions_B[-1])
        num_decisions += 1

        if repair_downtime_remaining_A == 0 and idle_count_A >= maintenance_downtime:
            #print "############ EXCAVATOR-A MAINTAINED  #####################"
            actions_performed_without_maintenance_A = 0
            idle_count_A = 0
        if repair_downtime_remaining_B == 0 and idle_count_B >= maintenance_downtime:
            #print "############ EXCAVATOR-B MAINTAINED  #####################"
            actions_performed_without_maintenance_B = 0
            idle_count_B = 0

        if repair_downtime_remaining_A == 0 and actions_performed_without_maintenance_A >= actions_to_failure_without_maintenance:
            #print "############ EXCAVATOR-A FAILED  #####################"
            repair_downtime_remaining_A = repair_downtime

        if repair_downtime_remaining_B == 0 and actions_performed_without_maintenance_B >= actions_to_failure_without_maintenance:
            #print "############ EXCAVATOR-B FAILED  #####################"
            repair_downtime_remaining_B = repair_downtime

    return action


#input global params
SoilAmt = 2000000
TrckCst = 48
ExcCst = 65
OHCst = 75

#output global params (results / costs)
HourlyCst = []
Hrs = []
ProdRate = []
UnitCst = []
Mean_TD_Error = []
Iterations = 0 #number of decision iterations in an episode

def run_sim(nTrucks, BucketA_capacity, BucketB_capacity, Truck1_capacity, Truck2_capacity, Truck1_speedRatio, Truck2_speedRatio):
    global state
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

    #initial state
    state[0] = (nTrucks/4)
    state[1] = nTrucks/4
    state[2] = nTrucks/4
    state[3] = (nTrucks/4)

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
            env.process(EnterAreaA(env, 'truck%d' % i, TrkWtLdA, TrkWtLdB,
            ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
            ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))
        else:
            #truck moving process for each truck
            env.process(EnterAreaB(env, 'truck%d' % i, TrkWtLdA, TrkWtLdB,
            ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
            ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

    #monitoring process
    proc = env.process(monitor(env, DmpdSoil, TrkWtLdA, TrkWtLdB, SoilAmt, nTrucks, max(Truck1_capacity, Truck2_capacity)))
    #run all processes
    env.run(until=proc)

#main
def main():
    global num_of_load
    global num_of_dump
    global num_of_return
    global state
    global old_state
    global old_time
    global Mean_TD_Error
    global Iterations
    global nTrucks
    global num_decisions
    global num_decisions_A
    global num_decisions_B
    global local_decisions_A
    global local_decisions_B
    global idle_count_A
    global idle_count_B
    global actions_performed_without_maintenance_A
    global actions_performed_without_maintenance_B
    global repair_downtime_remaining_A
    global repair_downtime_remaining_B
    global both_excavators_failed_flag

    global g_Truck1_capacity
    global g_Truck2_capacity

    BucketA_capacity = 6
    BucketB_capacity = 3
    Truck1_capacity = g_Truck1_capacity
    Truck2_capacity = g_Truck2_capacity
    Truck1_speed = 20
    Truck2_speed = 20
    Truck1_speedRatio = Truck1_speed / float(Truck1_speed + Truck2_speed)
    Truck2_speedRatio = Truck2_speed / float(Truck1_speed + Truck2_speed)

    #run session (initialise tf global vars)
    sess.run(init)

    num_episodes = 1
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=[],
        episode_rewards=[],
        episode_loss=[],
        episode_decisions_A=[],
        episode_decisions_B=[],
        lastep_decisions_A=[],
        lastep_decisions_B=[])

    for i_episode in range(num_episodes):
        #reset global vars
        num_of_load = 0
        num_of_dump = 0
        num_of_return = 0
        state = np.zeros(12)
        old_state = np.zeros((nTrucks,12))
        old_time = np.zeros(nTrucks)
        Mean_TD_Error = []
        Iterations = 0
        num_decisions = 0
        num_decisions_A = []
        num_decisions_B = []
        idle_count_A = 0
        idle_count_B = 0
        actions_performed_without_maintenance_A = 0
        actions_performed_without_maintenance_B = 0
        repair_downtime_remaining_A = 0
        repair_downtime_remaining_B = 0
        both_excavators_failed_flag = False

        # Print out which episode we're on, useful for debugging.
        print "\rEpisode: ", i_episode + 1, " / ", num_episodes
        #run simulation
        run_sim(nTrucks, BucketA_capacity, BucketB_capacity, Truck1_capacity, Truck2_capacity, Truck1_speedRatio, Truck2_speedRatio)
        stats.episode_lengths.extend(Hrs)
        stats.episode_rewards.extend(ProdRate)
        stats.episode_loss.extend(Mean_TD_Error)
        stats.episode_decisions_A.extend(num_decisions_A)
        stats.episode_decisions_B.extend(num_decisions_B)

    stats.lastep_decisions_A.extend(local_decisions_A)
    stats.lastep_decisions_B.extend(local_decisions_B)
    # print "local_decisions_A: ", local_decisions_A
    # print "stats.lastep_decisions_A: ", stats.lastep_decisions_A
    # print stats.lastep_decisions_A == local_decisions_A

    #plotting.plot_episode_stats(stats, name='Qlearning_20_onehot_onebig_nn', smoothing_window=20)
    plotting.plot_episode_stats(stats, name='Qlearning_20_onehot_onebig_linear', smoothing_window=20)

if __name__ == '__main__':
    main()
