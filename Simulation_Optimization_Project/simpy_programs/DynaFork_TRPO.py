import simpy
import numpy as np
import tensorflow as tf
import scipy
from collections import defaultdict
import sys
import os
import itertools
import matplotlib

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

def DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity):
    global state
    while TrkUndrExcA.level == 0:
        yield env.timeout(1)
    #print "In DumpBucketA, TrkUndrExcA=%d" % TrkUndrExcA.level
    yield ExcWtDmpA.get(1)
    #print "starting DumpBucketA at %.2f" % env.now
    yield env.timeout(DumpBucketA_time())
    yield SlInTrkA.put(BucketA_capacity)
    if state[8] > 0:
        state[8] -= (BucketA_capacity/6)
    else:
        state[9] -= (BucketA_capacity/3)
    env.process(ExcavateA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity))
    #print "finished DumpBucketA with SlInTrkA=%.2f at %.2f" % (SlInTrkA.level, env.now)

def DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity):
    global state
    while TrkUndrExcB.level == 0:
        yield env.timeout(1)
    #print "In DumpBucketB, TrkUndrExcB=%d" % TrkUndrExcB.level
    yield ExcWtDmpB.get(1)
    #print "starting DumpBucketB at %.2f" % env.now
    yield env.timeout(DumpBucketB_time())
    yield SlInTrkB.put(BucketB_capacity)
    if state[10] > 0:
        state[10] -= (BucketB_capacity/6)
    else:
        state[11] -= (BucketB_capacity/3)
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
    #while (DmpdSoil.level + soil_ready_to_dump) > (SoilAmt - TruckCap): #wait here if SoilAmt met
    #    yield env.timeout(1)
    yield TrkUndrExcA.put(1) #will move forward only if TrkUndrExc.level was 0, also blocks other trucks
    #print "%s inside EnterAreaA - put 1 TrkUndrExcA. TrkUndrExcA.level=%d at %.2f" % (name, TrkUndrExcA.level, env.now)
    yield ManeuvSpcA.get(1)
    #print "%s inside EnterAreaA - got 1 ManeuvSpcA. ManeuvSpcA.level=%d at %.2f" % (name, ManeuvSpcA.level, env.now)
    yield TrkWtLdA.get(1)
    #print "%s starting EnterAreaA at %.2f" % (name, env.now)
    if TruckCap == 6:
        state[0] -= 1
        state[4] = 1
    else:
        state[1] -= 1
        state[5] = 1
    yield env.timeout(EnterAreaA_time(TruckSpdRatio))
    if TruckCap == 6:
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
    #while (DmpdSoil.level + soil_ready_to_dump) > (SoilAmt - TruckCap): #wait here if SoilAmt met
    #    yield env.timeout(1)
    yield TrkUndrExcB.put(1) #will move forward only if TrkUndrExc.level was 0, also blocks other trucks
    #print "%s inside EnterAreaB - put 1 TrkUndrExcB. TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcB.level, env.now)
    yield ManeuvSpcB.get(1)
    #print "%s inside EnterAreaB - got 1 ManeuvSpcB. ManeuvSpcB.level=%d at %.2f" % (name, ManeuvSpcB.level, env.now)
    yield TrkWtLdB.get(1)
    #print "%s starting EnterAreaB at %.2f" % (name, env.now)
    if TruckCap == 6:
        state[2] -= 1
        state[6] = 1
    else:
        state[3] -= 1
        state[7] = 1
    yield env.timeout(EnterAreaB_time(TruckSpdRatio))
    if TruckCap == 6:
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

    yield SlInTrkA.get(TruckCap)
    yield TrkUndrExcA.get(1)
    #print "%s inside Haul - got 1 TrkUndrExc. TrkUndrExcA.level=%d; TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcA.level, TrkUndrExcB.level, env.now)
    #print "-----%s starting Haul at %.2f" % (name, env.now)
    num_of_load += 1
    if TruckCap == 6:
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

    yield SlInTrkB.get(TruckCap)
    yield TrkUndrExcB.get(1)
    #print "%s inside Haul - got 1 TrkUndrExc. TrkUndrExcA.level=%d; TrkUndrExcB.level=%d at %.2f" % (name, TrkUndrExcA.level, TrkUndrExcB.level, env.now)
    #print "-----%s starting Haul at %.2f" % (name, env.now)
    num_of_load += 1
    if TruckCap == 6:
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
    #print "%s starting Return0 at %.2f" % (name, env.now)
    yield env.timeout(Return0_time(TruckSpdRatio))
    #print "%s finished Return0 at %.2f" % (name, env.now)
    action = agent(name, env.now)
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
    #print "%s starting Return1A at %.2f" % (name, env.now)
    yield env.timeout(Return1A_time(TruckSpdRatio))
    #print "-----%s finished Return1A at %.2f" % (name, env.now)
    yield TrkWtLdA.put(1)
    if TruckCap == 6:
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
    #print "%s starting Return1B at %.2f" % (name, env.now)
    yield env.timeout(Return1B_time(TruckSpdRatio))
    #print "-----%s finished Return1B at %.2f" % (name, env.now)
    yield TrkWtLdB.put(1)
    if TruckCap == 6:
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
        if L_hrs > 0:
            L_prodRate = DmpdSoil.level/L_hrs
            if L_prodRate > 0:
                L_unitCst = L_hourlyCst/L_prodRate

        #terminate condition
        if (DmpdSoil.level > (SoilAmt-TruckCap)):
            #print "\nDmpdSoil.level = %d. Desired SoilAmt achieved" % DmpdSoil.level

            #update global stats
            Hrs.append(L_hrs)
            HourlyCst.append(L_hourlyCst)
            ProdRate.append(L_prodRate)
            UnitCst.append(L_unitCst)

            print """
            nTrucks = %d\n
            num_of_load = %d \n
            num_of_dump = %d \n
            num_of_return = %d \n
            Hrs = %.4f \n
            HourlyCst = %.4f \n
            ProdRate = %.4f \n
            UnitCst = %.4f
            """ % (nTrucks, num_of_load, num_of_dump, num_of_return,
            L_hrs, L_hourlyCst, L_prodRate, L_unitCst)
            return

        yield env.timeout(1)


#seed
np.random.seed(0)

#global variables
num_of_load = 0
num_of_dump = 0
num_of_return = 0
nTrucks = 10

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
state[11]: % empty Trk2 UndrExcB '''
state = np.zeros(12)
old_state = np.zeros((nTrucks,12))
old_time = np.zeros(nTrucks)
old_action = np.zeros(nTrucks).astype(int)
nA = 2 #number of actions
old_action_probs = np.zeros(nA)
discount_factor = 0.99
alpha_vf = 1e-3 #learning_rate
alpha_policy = 1e-4 #learning_rate
kl_target = 0.003 #max KL divergence allowed
max_policy_epochs = 20

#get interactive session
sess = tf.InteractiveSession()
#placeholder for observation
obs = tf.placeholder(tf.float32, shape=[12])

#vf network
fc_shared = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(obs,0), num_outputs=24, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
vf_output = tf.contrib.layers.fully_connected(inputs=fc_shared, num_outputs=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
#state value
state_value = tf.squeeze(vf_output)
#vf target
vf_target = tf.placeholder(tf.float32, shape=[])
#vf loss
vf_loss = tf.squared_difference(vf_target, state_value)
#optimizer
vf_optimizer = tf.train.AdamOptimizer(learning_rate=alpha_vf)
#training op
vf_train_op = vf_optimizer.minimize(vf_loss)

#policy network
policy_hidden = tf.contrib.layers.fully_connected(inputs=fc_shared, num_outputs=12, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
policy_output = tf.contrib.layers.fully_connected(inputs=policy_hidden, num_outputs=nA, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
#action probabilities
action_probs = tf.squeeze(tf.nn.softmax(policy_output))
#old policy distribution
action_probs_old = tf.placeholder(tf.float32, shape=[2])
#placeholder for chosen action
chosen_action = tf.placeholder(tf.int32, shape=[])
#probability of chosen action
prob_chosen_action = tf.gather(action_probs, chosen_action)
#probability of chosen action in old policy
oldprob_chosen_action = tf.gather(action_probs_old, chosen_action)
#placeholder for advantage
advantage = tf.placeholder(tf.float32, shape=[])
#surrogate loss function - TODO stop gradient on advantage ?
surrloss = -tf.reduce_mean(advantage * (prob_chosen_action / oldprob_chosen_action))
#KL divergence between old policy and current policy - TODO iterated learning on same obs
kldiv = tf.reduce_sum(action_probs_old * tf.log((action_probs_old + 1e-10) / (action_probs + 1e-10)))
#entropy of current policy
entropy = tf.reduce_sum(-action_probs * tf.log(action_probs + 1e-10))
#Experiment: add entropy to current loss
final_loss = tf.add(surrloss, 0.1*entropy)
#optimizer
policy_optimizer = tf.train.AdamOptimizer(learning_rate=alpha_policy)
#training op
policy_train_op = policy_optimizer.minimize(final_loss)

#global vars init
init = tf.global_variables_initializer()

#Q-learning for now
def agent(truckName, time):
    global state #current state treated as next state
    global old_state #state for which we are learning
    global old_time
    global old_action
    global old_action_probs
    global Mean_TD_Error
    global Iterations
    global kl_target
    global max_policy_epochs
    truckIndex = int(truckName[len('truck')])
    reward = -1 * (time - old_time[truckIndex]) #time of cycle for this truck

    if old_time[truckIndex] > 0:  #not the first ever decision - learn for the old_state
        cur_state_value = state_value.eval(feed_dict={obs: state})
        old_state_value = state_value.eval(feed_dict={obs: old_state[truckIndex]})
        td_target = reward + discount_factor * cur_state_value
        td_error = td_target - old_state_value
        Iterations += 1
        Mean_TD_Error = ((Iterations-1)*Mean_TD_Error + td_error)/Iterations
        #update
        sess.run(vf_train_op, feed_dict={obs: old_state[truckIndex], vf_target: td_target})
        policy_epochs = 0
        while policy_epochs < max_policy_epochs:
            sess.run(policy_train_op, feed_dict={obs: old_state[truckIndex], chosen_action: old_action[truckIndex], advantage: td_error, action_probs_old: old_action_probs})
            kl = kldiv.eval(feed_dict={obs: old_state[truckIndex], action_probs_old: old_action_probs})
            if kl > kl_target:
                break
            policy_epochs += 1
        #get action for current state
        cur_action_probs = action_probs.eval(feed_dict={obs: state})
        action = np.random.choice(np.arange(nA), p=cur_action_probs)
    else: #first ever decision - no learning since no old_action
        cur_action_probs = action_probs.eval(feed_dict={obs: state})
        action = np.random.choice(np.arange(nA), p=cur_action_probs)
        Iterations = 1
    #set up for next decision call
    np.copyto(old_state[truckIndex], state)
    old_action[truckIndex] = action
    old_action_probs = cur_action_probs
    old_time[truckIndex] = time

    return action


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
Mean_TD_Error = 0
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
    state[0] = (nTrucks/4) + 1
    state[1] = nTrucks/4
    state[2] = nTrucks/4
    state[3] = (nTrucks/4) + 1

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

    BucketA_capacity = 1.5
    BucketB_capacity = 1.0
    Truck1_capacity = 6
    Truck2_capacity = 3
    Truck1_speed = 15.0
    Truck2_speed = 20.0
    Truck1_speedRatio = Truck1_speed / (Truck1_speed + Truck2_speed)
    Truck2_speedRatio = Truck2_speed / (Truck1_speed + Truck2_speed)

    #run session (initialise tf global vars)
    sess.run(init)

    num_episodes = 400
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_loss=np.zeros(num_episodes))
    for i_episode in range(num_episodes):
        #reset global vars
        num_of_load = 0
        num_of_dump = 0
        num_of_return = 0
        state = np.zeros(12)
        old_state = np.zeros((nTrucks,12))
        old_time = np.zeros(nTrucks)
        Mean_TD_Error = 0
        Iterations = 0
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1 == 0:
            print "\rEpisode: ", i_episode + 1, " / ", num_episodes
        #run simulation
        run_sim(nTrucks, BucketA_capacity, BucketB_capacity, Truck1_capacity, Truck2_capacity, Truck1_speedRatio, Truck2_speedRatio)
        stats.episode_lengths[i_episode] = Hrs[i_episode]
        stats.episode_rewards[i_episode] = ProdRate[i_episode]
        stats.episode_loss[i_episode] = abs(Mean_TD_Error)
    plotting.plot_episode_stats(stats, smoothing_window=20)

if __name__ == '__main__':
    main()
