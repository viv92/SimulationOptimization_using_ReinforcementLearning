import simpy
import numpy as np
import tensorflow as tf
import scipy
from collections import defaultdict
import sys
import os
import itertools
import matplotlib
import threading
import multiprocessing
import time

from lib import plotting
matplotlib.style.use('ggplot')

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

def DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity, i_episode):
    global state
    while TrkUndrExcA.level == 0:
        yield env.timeout(1)
    #print "In DumpBucketA, TrkUndrExcA=%d" % TrkUndrExcA.level
    yield ExcWtDmpA.get(1)
    #print "starting DumpBucketA at %.2f" % env.now
    yield env.timeout(DumpBucketA_time())
    yield SlInTrkA.put(BucketA_capacity)
    if state[i_episode][8] > 0:
        state[i_episode][8] -= (BucketA_capacity/6)
    else:
        state[i_episode][9] -= (BucketA_capacity/3)
    env.process(ExcavateA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity, i_episode))
    #print "finished DumpBucketA with SlInTrkA=%.2f at %.2f" % (SlInTrkA.level, env.now)

def DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity, i_episode):
    global state
    while TrkUndrExcB.level == 0:
        yield env.timeout(1)
    #print "In DumpBucketB, TrkUndrExcB=%d" % TrkUndrExcB.level
    yield ExcWtDmpB.get(1)
    #print "starting DumpBucketB at %.2f" % env.now
    yield env.timeout(DumpBucketB_time())
    yield SlInTrkB.put(BucketB_capacity)
    if state[i_episode][10] > 0:
        state[i_episode][10] -= (BucketB_capacity/6)
    else:
        state[i_episode][11] -= (BucketB_capacity/3)
    env.process(ExcavateB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity, i_episode))
    #print "finished DumpBucketB with SlInTrkB=%.2f at %.2f" % (SlInTrkB.level, env.now)

def ExcavateA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity, i_episode):
    #print "starting ExcavateA at %.2f" % env.now
    yield env.timeout(ExcavateA_time())
    #print "finished ExcavateA at %.2f" % env.now
    yield ExcWtDmpA.put(1)
    env.process(DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity, i_episode))
    #print "finished ExcavateA at %.2f" % env.now

def ExcavateB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity, i_episode):
    #print "starting ExcavateB at %.2f" % env.now
    yield env.timeout(ExcavateB_time())
    #print "finished ExcavateB at %.2f" % env.now
    yield ExcWtDmpB.put(1)
    env.process(DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity, i_episode))

def EnterAreaA(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
    global state
    #while (DmpdSoil.level + soil_ready_to_dump) > (SoilAmt - TruckCap): #wait here if SoilAmt met
    #    yield env.timeout(1)
    yield TrkUndrExcA.put(1) #will move forward only if TrkUndrExc.level was 0, also blocks other trucks
    #print "%s inside EnterAreaA - put 1 TrkUndrExcA. TrkUndrExcA.level=%d at %.2f" % (name, TrkUndrExcA.level, env.now)
    yield ManeuvSpcA.get(1)
    #print "%s inside EnterAreaA - got 1 ManeuvSpcA. ManeuvSpcA.level=%d at %.2f" % (name, ManeuvSpcA.level, env.now)
    yield TrkWtLdA.get(1)
    #print "%s starting EnterAreaA at %.2f" % (name, env.now)
    if TruckCap == 6:
        state[i_episode][0] -= 1
        state[i_episode][4] = 1
    else:
        state[i_episode][1] -= 1
        state[i_episode][5] = 1
    yield env.timeout(EnterAreaA_time(TruckSpdRatio))
    if TruckCap == 6:
        state[i_episode][4] = 0
        state[i_episode][8] = 1
    else:
        state[i_episode][5] = 0
        state[i_episode][9] = 1
    #print "-----%s finished EnterAreaA at %.2f" % (name, env.now)
    yield ManeuvSpcA.put(1)
    #print "%s post EnterAreaA - put 1 ManeuvSpcA. ManeuvSpcA.level=%d at %.2f" % (name, ManeuvSpcA.level, env.now)
    env.process(HaulA(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))

def EnterAreaB(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
    global state
    #while (DmpdSoil.level + soil_ready_to_dump) > (SoilAmt - TruckCap): #wait here if SoilAmt met
    #    yield env.timeout(1)
    yield TrkUndrExcB.put(1) #will move forward only if TrkUndrExc.level was 0, also blocks other trucks
    #print "%s inside EnterAreaB - put 1 TrkUndrExcB. TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcB.level, env.now)
    yield ManeuvSpcB.get(1)
    #print "%s inside EnterAreaB - got 1 ManeuvSpcB. ManeuvSpcB.level=%d at %.2f" % (name, ManeuvSpcB.level, env.now)
    yield TrkWtLdB.get(1)
    #print "%s starting EnterAreaB at %.2f" % (name, env.now)
    if TruckCap == 6:
        state[i_episode][2] -= 1
        state[i_episode][6] = 1
    else:
        state[i_episode][3] -= 1
        state[i_episode][7] = 1
    yield env.timeout(EnterAreaB_time(TruckSpdRatio))
    if TruckCap == 6:
        state[i_episode][6] = 0
        state[i_episode][10] = 1
    else:
        state[i_episode][7] = 0
        state[i_episode][11] = 1
    #print "-----%s finished EnterAreaB at %.2f" % (name, env.now)
    yield ManeuvSpcB.put(1)
    #print "%s post EnterAreaB - put 1 ManeuvSpcB. ManeuvSpcB.level=%d at %.2f" % (name, ManeuvSpcB.level, env.now)
    env.process(HaulB(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))

def HaulA(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
    global state
    global num_of_load

    yield SlInTrkA.get(TruckCap)
    yield TrkUndrExcA.get(1)
    #print "%s inside Haul - got 1 TrkUndrExc. TrkUndrExcA.level=%d; TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcA.level, TrkUndrExcB.level, env.now)
    #print "-----%s starting Haul at %.2f" % (name, env.now)
    num_of_load[i_episode] += 1
    if TruckCap == 6:
        state[i_episode][8] = 0
    else:
        state[i_episode][9] = 0
    yield env.timeout(Haul_time(TruckSpdRatio))
    #print "%s finished Haul at %.2f" % (name, env.now)
    yield WtEnterDump.put(1)
    #print "%s post Haul - put 1 WtEnterDump. WtEnterDump.level=%d at %.2f" % (name, WtEnterDump.level, env.now)
    env.process(EnterDump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))

def HaulB(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
    global state
    global num_of_load

    yield SlInTrkB.get(TruckCap)
    yield TrkUndrExcB.get(1)
    #print "%s inside Haul - got 1 TrkUndrExc. TrkUndrExcA.level=%d; TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcA.level, TrkUndrExcB.level, env.now)
    #print "-----%s starting Haul at %.2f" % (name, env.now)
    num_of_load[i_episode] += 1
    if TruckCap == 6:
        state[i_episode][10] = 0
    else:
        state[i_episode][11] = 0
    yield env.timeout(Haul_time(TruckSpdRatio))
    #print "%s finished Haul at %.2f" % (name, env.now)
    yield WtEnterDump.put(1)
    #print "%s post Haul - put 1 WtEnterDump. WtEnterDump.level=%d at %.2f" % (name, WtEnterDump.level, env.now)
    env.process(EnterDump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))


def EnterDump(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
    yield DumpSpots.get(1)
    #print "%s Inside EnterDump - got 1 DumpSpots. DumpSpots.level=%d at %.2f" % (name, DumpSpots.level, env.now)
    yield WtEnterDump.get(1)
    #print "%s Inside EnterDump - got 1 WtEnterDump. WtEnterDump.level=%d at %.2f" % (name, WtEnterDump.level, env.now)
    #print "%s starting EnterDump at %.2f" % (name, env.now)
    yield env.timeout(EnterDump_time(TruckSpdRatio))
    #print "%s finished EnterDump at %.2f" % (name, env.now)
    env.process(Dump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))


def Dump(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
    global num_of_dump
    #print "%s starting Dump at %.2f" % (name, env.now)
    yield env.timeout(Dump_time())
    #print "%s finished Dump at %.2f" % (name, env.now)
    yield DmpdSoil.put(TruckCap)
    #print "%s post Dump - put %d DmpdSoil. DmpdSoil.level=%d at %.2f" % (name, TruckCap, DmpdSoil.level, env.now)
    yield DumpSpots.put(1)
    num_of_dump[i_episode] += 1
    #print "%s post Dump - put 1 DumpSpots. DumpSpots.level=%d at %.2f" % (name, DumpSpots.level, env.now)
    env.process(Return0(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))


def Return0(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
    global num_of_return
    #print "%s starting Return0 at %.2f" % (name, env.now)
    yield env.timeout(Return0_time(TruckSpdRatio))
    #print "%s finished Return0 at %.2f" % (name, env.now)
    action, ol_state, ol_action, ol_time, Iter, Mean_TD_Err = worker.work(name, env.now, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess)
    #print "*** action decided ***"
    if action == 0:
        env.process(Return1A(env, name, TrkWtLdA, TrkWtLdB,
        ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
        ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, ol_state, ol_time, ol_action, Iter, Mean_TD_Err, i_episode, coord, sess))
    else:
        env.process(Return1B(env, name, TrkWtLdA, TrkWtLdB,
        ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
        ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, ol_state, ol_time, ol_action, Iter, Mean_TD_Err, i_episode, coord, sess))
    num_of_return[i_episode] += 1


def Return1A(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
    global state
    #print "%s starting Return1A at %.2f" % (name, env.now)
    yield env.timeout(Return1A_time(TruckSpdRatio))
    #print "-----%s finished Return1A at %.2f" % (name, env.now)
    yield TrkWtLdA.put(1)
    if TruckCap == 6:
        state[i_episode][0] += 1
    else:
        state[i_episode][1] += 1
    #print "%s post Return1A - put 1 TrkWtLdA. TrkWtLdA.level=%d at %.2f" % (name, TrkWtLdA.level, env.now)
    env.process(EnterAreaA(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))

def Return1B(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
    global state
    #print "%s starting Return1B at %.2f" % (name, env.now)
    yield env.timeout(Return1B_time(TruckSpdRatio))
    #print "-----%s finished Return1B at %.2f" % (name, env.now)
    yield TrkWtLdB.put(1)
    if TruckCap == 6:
        state[i_episode][2] += 1
    else:
        state[i_episode][3] += 1
    #print "%s post Return1B - put 1 TrkWtLdB. TrkWtLdB.level=%d at %.2f" % (name, TrkWtLdB.level, env.now)
    env.process(EnterAreaB(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))

#define monitoring process
def monitor(env, DmpdSoil, TrkWtLdA, TrkWtLdB, SoilAmt, nTrucks, TruckCap, i_episode):
    #global output containers
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
            #print "\nDmpdSoil.level = %d. Desired SoilAmt achieved" % DmpdSoil.level

            #calculate outputs
            L_hrs = env.now/60.0
            L_hourlyCst = OHCst+ExcCst+(TrckCst*nTrucks) #duh constant
            if L_hrs > 0:
                L_prodRate = DmpdSoil.level/L_hrs
                if L_prodRate > 0:
                    L_unitCst = L_hourlyCst/L_prodRate

            #update global stats
            Hrs[i_episode] = L_hrs
            HourlyCst[i_episode] = L_hourlyCst
            ProdRate[i_episode] = L_prodRate
            UnitCst[i_episode] = L_unitCst

            print """
            nTrucks = %d\n
            num_of_load = %d \n
            num_of_dump = %d \n
            num_of_return = %d \n
            Hrs = %.4f \n
            HourlyCst = %.4f \n
            ProdRate = %.4f \n
            UnitCst = %.4f
            """ % (nTrucks, num_of_load[i_episode], num_of_dump[i_episode], num_of_return[i_episode],
            L_hrs, L_hourlyCst, L_prodRate, L_unitCst)
            return

        yield env.timeout(1)

#global variables
nTrucks = 10
nSteps = 5
num_episodes = 200
nA = 2 #number of actions
nS = 12 #size of state vector
discount_factor = 0.99
alpha_critic = 9*1e-6 #learning_rate
alpha_actor = 9*1e-7 #learning_rate
#global input params
SoilAmt = 10000
TrckCst = 48
ExcCst = 65
OHCst = 75
state = np.zeros((num_episodes, nS))
#global output params (containers for result)
HourlyCst = np.zeros(num_episodes)
Hrs = np.zeros(num_episodes)
ProdRate = np.zeros(num_episodes)
UnitCst = np.zeros(num_episodes)
Mean_Loss = np.zeros(num_episodes)
num_of_load = np.zeros(num_episodes)
num_of_dump = np.zeros(num_episodes)
num_of_return = np.zeros(num_episodes)

#helper function to copy global network weights to a worker network
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    #variable to hold assign ops
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

#class for actor-critic network
class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer_critic, trainer_actor):
        with tf.variable_scope(scope):
            #define network

            #placeholder for observation
            self.obs = tf.placeholder(tf.float32, shape=[s_size])
            #critic network
            self.fc_shared = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.obs,0), num_outputs=24, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            self.critic_output = tf.contrib.layers.fully_connected(inputs=self.fc_shared, num_outputs=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            #state value
            self.state_value = tf.squeeze(self.critic_output)

            #actor network
            self.actor_hidden = tf.contrib.layers.fully_connected(inputs=self.fc_shared, num_outputs=12, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
            self.actor_output = tf.contrib.layers.fully_connected(inputs=self.actor_hidden, num_outputs=a_size, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            #action probabilities
            self.action_probs = tf.squeeze(tf.nn.softmax(self.actor_output))

            #only worker needs ops for loss function and gradient updating
            if scope != 'global':
                #critic target
                self.critic_target = tf.placeholder(tf.float32, shape=[])
                #critic loss
                self.critic_loss = tf.squared_difference(self.critic_target, self.state_value)
                #placeholder for chosen action
                self.chosen_action = tf.placeholder(tf.int32, shape=[])
                #probability of chosen action
                self.prob_chosen_action = tf.gather(self.action_probs, self.chosen_action)
                #placeholder for advantage
                self.advantage = tf.placeholder(tf.float32, shape=[])
                #loss function
                self.loss = -tf.reduce_sum(tf.log(self.prob_chosen_action + 1e-10) * tf.stop_gradient(self.advantage))
                #self.loss = -tf.reduce_sum(tf.log(self.prob_chosen_action) * tf.stop_gradient(self.advantage))
                #add entropy of action_probs to encourage exploration
                self.entropy = tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 1e-10))
                #self.entropy = tf.reduce_sum(self.action_probs * tf.log(self.action_probs))
                #final loss
                self.actor_loss = tf.add(self.loss, 0.1*self.entropy)

                #get gradients from local network using local losses
                local_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                #TODO combine critic loss and actor loss ?
                self.gradients_critic = tf.gradients(self.critic_loss, local_trainable_vars)
                self.gradients_actor = tf.gradients(self.actor_loss, local_trainable_vars)
                #self.var_norms = tf.global_norm(local_trainable_vars)
                #TODO adjust gradient clipping threshold
                #grads_critic, self.grad_norms_critic = tf.clip_by_global_norm(self.gradients_critic, 40.0)
                #grads_actor, self.grad_norms_actor = tf.clip_by_global_norm(self.gradients_actor, 40.0)

                #apply local gradients to global network
                global_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                #self.apply_grads_critic = trainer_critic.apply_gradients(zip(grads_critic, global_trainable_vars))
                self.apply_grads_critic = trainer_critic.apply_gradients(zip(self.gradients_critic, global_trainable_vars))
                #self.apply_grads_actor = trainer_actor.apply_gradients(zip(grads_actor, global_trainable_vars))
                self.apply_grads_actor = trainer_actor.apply_gradients(zip(self.gradients_actor, global_trainable_vars))

#class for worker
class Worker():
    def __init__(self, name, s_size, a_size, trainer_critic, trainer_actor):
        self.name = "worker_" + str(name)
        self.number = name
        self.trainer_critic = trainer_critic
        self.trainer_actor = trainer_actor
        #create local copy of network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer_critic, trainer_actor)
        #op to copy global network weights to local network
        self.update_local_ops = update_target_graph('global', self.name)
        #episode buffer and n-step counter
        global nTrucks
        self.s_buf = np.zeros(((nTrucks, nSteps, nS)))
        self.a_buf = np.zeros((nTrucks, nSteps))
        self.r_buf = np.zeros((nTrucks, nSteps))
        self.step_counter = np.zeros(nTrucks)

    def work(self, truckName, time, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
        global state
        global Mean_Loss
        global nSteps
        #print "Starting worker: ", self.name
        with sess.as_default(), sess.graph.as_default():
            #print "Default graph: ", tf.get_default_graph()
            while not coord.should_stop():
                #update wieghts of local network to match weights of global network
                #sess.run(self.update_local_ops)
                #TODO self.truckIndex ?
                truckIndex = int(truckName[len('truck')])

                if old_time[truckIndex] > 0: #not the first ever decision
                    reward = -1 * (time - old_time[truckIndex]) #time of cycle for this truck
                    #populate episode buffer for reward and increment step counter
                    self.r_buf[truckIndex][self.step_counter[truckIndex]] = reward
                    self.step_counter[truckIndex] += 1

                    if self.step_counter[truckIndex] == nSteps: #time to learn and update params
                        #make local copy of step_counter for ease
                        stepIndex = self.step_counter[truckIndex] - 1
                        #bootstrap for n-th state
                        G = self.local_AC.state_value.eval(feed_dict={self.local_AC.obs: state[i_episode]}, session=sess)
                        while stepIndex >= 0:
                            #state value for state denoted by stepIndex
                            v = self.local_AC.state_value.eval(feed_dict={self.local_AC.obs: self.s_buf[truckIndex][stepIndex]}, session=sess)
                            #bootstrap for state denoted by stepIndex
                            G = self.r_buf[truckIndex][stepIndex] + (discount_factor * G)
                            error = G - v
                            Iterations += 1
                            Mean_TD_Error = ((Iterations-1)*Mean_TD_Error + error)/Iterations
                            Mean_Loss[i_episode] = abs(Mean_TD_Error)
                            #train global network
                            sess.run(self.local_AC.apply_grads_critic, feed_dict={self.local_AC.obs: self.s_buf[truckIndex][stepIndex], self.local_AC.critic_target: G})
                            sess.run(self.local_AC.apply_grads_actor, feed_dict={self.local_AC.obs: self.s_buf[truckIndex][stepIndex], self.local_AC.chosen_action: self.a_buf[truckIndex][stepIndex], self.local_AC.advantage: error})
                            #go to previous step and repeat
                            stepIndex -= 1

                        #global network updated. Reset episode buffers
                        self.step_counter[truckIndex] = 0
                        self.s_buf[truckIndex] = np.zeros((nSteps, nS))
                        self.a_buf[truckIndex] = np.zeros(nSteps)
                        self.r_buf[truckIndex] = np.zeros(nSteps)
                        #match wieghts of local network to weights of global network
                        sess.run(self.update_local_ops)
                        #get action for current state
                        cur_action_probs = self.local_AC.action_probs.eval(feed_dict={self.local_AC.obs: state[i_episode]}, session=sess)
                        action = np.random.choice(np.arange(nA), p=cur_action_probs)

                    else: #not-learning (just behaving)
                        cur_action_probs = self.local_AC.action_probs.eval(feed_dict={self.local_AC.obs: state[i_episode]}, session=sess)
                        action = np.random.choice(np.arange(nA), p=cur_action_probs)
                        #Iterations += 1

                else: #first ever decision
                    cur_action_probs = self.local_AC.action_probs.eval(feed_dict={self.local_AC.obs: state[i_episode]}, session=sess)
                    action = np.random.choice(np.arange(nA), p=cur_action_probs)
                    Iterations = 1

                #set up for next decision call
                np.copyto(old_state[truckIndex], state[i_episode])
                old_action[truckIndex] = action
                old_time[truckIndex] = time
                #populate episode buffer for state and action
                np.copyto(self.s_buf[truckIndex][self.step_counter[truckIndex]], old_state[truckIndex])
                self.a_buf[truckIndex][self.step_counter[truckIndex]] = old_action[truckIndex]
                return action, old_state, old_action, old_time, Iterations, Mean_TD_Error


def run_sim(nTrucks, BucketA_capacity, BucketB_capacity, Truck1_capacity, Truck2_capacity, Truck1_speedRatio, Truck2_speedRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess):
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
    env.process(DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity, i_episode))
    env.process(DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity, i_episode))

    #initial state
    state[i_episode][0] = (nTrucks/4) + 1
    state[i_episode][1] = nTrucks/4
    state[i_episode][2] = nTrucks/4
    state[i_episode][3] = (nTrucks/4) + 1

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
            ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))
        else:
            #truck moving process for each truck
            env.process(EnterAreaB(env, 'truck%d' % i, TrkWtLdA, TrkWtLdB,
            ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
            ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess))

    #monitoring process
    proc = env.process(monitor(env, DmpdSoil, TrkWtLdA, TrkWtLdB, SoilAmt, nTrucks, max(Truck1_capacity, Truck2_capacity), i_episode))
    #run all processes
    env.run(until=proc)

#main
def main():

    global nTrucks
    global state
    global num_of_load
    global num_of_dump
    global num_of_return
    BucketA_capacity = 1.5
    BucketB_capacity = 1.0
    Truck1_capacity = 6
    Truck2_capacity = 3
    Truck1_speed = 15.0
    Truck2_speed = 20.0
    Truck1_speedRatio = Truck1_speed / (Truck1_speed + Truck2_speed)
    Truck2_speedRatio = Truck2_speed / (Truck1_speed + Truck2_speed)
    i_episode = 0

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_loss=np.zeros(num_episodes))

    #seed - TODO seed()
    np.random.seed(0)

    #initialize master and workers
    tf.reset_default_graph()
    with tf.device("/cpu:0"):
        trainer_critic = tf.train.AdamOptimizer(learning_rate=alpha_critic)
        trainer_actor = tf.train.AdamOptimizer(learning_rate=alpha_actor)
        master_network = AC_Network(nS, nA, 'global', None, None)
        #num_threads = multiprocessing.cpu_count() #TODO does each thread run on a different cpu core ?
        num_threads = 3
        workers = []
        #create workers
        for i in range(num_threads):
            workers.append(Worker(i, nS, nA, trainer_critic, trainer_actor))

    #set up session
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        #start episodes
        while (i_episode + num_threads) < num_episodes:
            #reset vars
            num_of_load = np.zeros(num_episodes)
            num_of_dump = np.zeros(num_episodes)
            num_of_return = np.zeros(num_episodes)
            state[i_episode] = np.zeros(nS)
            old_state = np.zeros((nTrucks,nS))
            old_time = np.zeros(nTrucks)
            old_action = np.zeros(nTrucks).astype(int)
            Iterations = 0 #number of decision iterations in an episode
            Mean_TD_Error = 0

            #initialize environment threads
            env_threads = []
            for worker in workers:
                #reset episode buffers
                worker.s_buf = np.zeros(((nTrucks, nSteps, nS)))
                worker.a_buf = np.zeros((nTrucks, nSteps))
                worker.r_buf = np.zeros((nTrucks, nSteps))
                worker.step_counter = np.zeros(nTrucks).astype(int)

                run_sim_args = [nTrucks, BucketA_capacity, BucketB_capacity, Truck1_capacity, Truck2_capacity, Truck1_speedRatio, Truck2_speedRatio, worker, old_state, old_time, old_action, Iterations, Mean_TD_Error, i_episode, coord, sess]
                # Print num of episode
                print "\rEpisode: ", i_episode + 1, " / ", num_episodes
                i_episode += 1
                t = threading.Thread(target=run_sim, args=run_sim_args)
                t.start()
                env_threads.append(t)
                #time.sleep(1)
            coord.join(env_threads)

        for i in range(num_episodes):
            if i >= i_episode:
                stats.episode_lengths[i] = Hrs[i_episode-1]
                stats.episode_rewards[i] = ProdRate[i_episode-1]
                stats.episode_loss[i] = Mean_Loss[i_episode-1]
                #print "Mean_Loss[%d] = %r \t eploss[%d] = %r" % (i_episode-1, Mean_Loss[i_episode-1], i, stats.episode_loss[i])
            else:
                stats.episode_lengths[i] = Hrs[i]
                stats.episode_rewards[i] = ProdRate[i]
                stats.episode_loss[i] = Mean_Loss[i]
                #print "Mean_Loss[%d] = %r \t eploss[%d] = %r" % (i, Mean_Loss[i], i, stats.episode_loss[i])
        plotting.plot_episode_stats(stats, name='A3C', smoothing_window=20)

if __name__ == '__main__':
    main()
