import simpy
import numpy as np
import scipy
import matplotlib.pyplot as plt
#import mcerp

np.random.seed(0)

num_of_load = 0
num_of_haul = 0
num_of_dump = 0
num_of_return = 0

def load_time():
    return np.random.uniform(1.3, 1.8)

def haul_time():
    return np.random.triangular(4, 5.5, 6)

def dump_time():
    return 0.5

def return_time():
    return np.random.triangular(3, 4, 5)

def load(env, name, truckContainer, soilInStkPl, whlLdr, soilInDump):
    global num_of_load
    yield whlLdr.get(1)
    yield soilInStkPl.get(15)
    yield truckContainer.get(1)
    print "%s starting to load at %.2f" % (name, env.now)
    yield env.timeout(load_time())
    yield whlLdr.put(1)
    env.process(haul(env, name, truckContainer, soilInStkPl, whlLdr, soilInDump))
    num_of_load += 1
    print "%s finished load at %.2f" % (name, env.now)

def haul(env, name, truckContainer, soilInStkPl, whlLdr, soilInDump):
    global num_of_haul
    print "%s starting to haul at %.2f" % (name, env.now)
    yield env.timeout(haul_time())
    env.process(dump(env, name, truckContainer, soilInStkPl, whlLdr, soilInDump))
    num_of_haul += 1
    print "%s finished haul at %.2f" % (name, env.now)

def dump(env, name, truckContainer, soilInStkPl, whlLdr, soilInDump):
    global num_of_dump
    print "%s starting to dump at %.2f" % (name, env.now)
    yield env.timeout(dump_time())
    yield soilInDump.put(15)
    env.process(returnback(env, name, truckContainer, soilInStkPl, whlLdr, soilInDump))
    num_of_dump += 1
    print "%s finished dump at %.2f" % (name, env.now)

def returnback(env, name, truckContainer, soilInStkPl, whlLdr, soilInDump):
    global num_of_return
    print "%s starting to return at %.2f" % (name, env.now)
    yield env.timeout(return_time())
    yield truckContainer.put(1)
    env.process(load(env, name, truckContainer, soilInStkPl, whlLdr, soilInDump))
    num_of_return += 1
    print "%s finished return at %.2f" % (name, env.now)

def monitor(env, soilInStkPl, truckContainer):
    global num_of_load
    global num_of_haul
    global num_of_dump
    global num_of_return

    obs_num = 0
    dur = np.zeros(6)
    prev_level = truckContainer.level
    prev_time = env.now

    while True:
        obs_num += 1
        print "\nobs_num=%d\t soilInStkPl=%d\t truckContainer=%d\n" % (
        obs_num, soilInStkPl.level, truckContainer.level)

        curr_level = truckContainer.level
        curr_time = env.now
        if prev_level != curr_level:
            duration = curr_time - prev_time
            dur[prev_level] += duration
            prev_level = curr_level
            prev_time = curr_time

        if (soilInStkPl.level < 15) and (truckContainer.level == 5):
            print "\nSoil in Stock Pile < 15"
            print """
            num_of_load = %d \n
            num_of_haul = %d \n
            num_of_dump = %d \n
            num_of_return = %d \n
            """ % (num_of_load, num_of_haul, num_of_dump, num_of_return)

            total_time = env.now
            dur0 = dur[0]
            dur1 = dur0 + dur[1]
            dur2 = dur1 + dur[2]
            dur3 = dur2 + dur[3]
            dur4 = dur[4] + dur[5]
            print "\n<1 \t %.2f \t %.2f" % (dur0, (dur0*100)/total_time)
            print "<2 \t %.2f \t %.2f" % (dur1, (dur1*100)/total_time)
            print "<3 \t %.2f \t %.2f" % (dur2, (dur2*100)/total_time)
            print "<4 \t %.2f \t %.2f" % (dur3, (dur3*100)/total_time)
            print ">=4 \t %.2f \t %.2f" % (dur4, (dur4*100)/total_time)
            return
        yield env.timeout(1)

env = simpy.Environment()

num_trucks = 5
truckContainer = simpy.Container(env, init=num_trucks, capacity=num_trucks)
soilInStkPl = simpy.Container(env, init=1000, capacity=1000)
soilInDump = simpy.Container(env, init=0, capacity=1000)
whlLdr = simpy.Container(env, init=1, capacity=1)

for i in range(num_trucks):
    name = "truck %d" % i
    env.process(load(env, name, truckContainer, soilInStkPl, whlLdr, soilInDump))

proc = env.process(monitor(env, soilInStkPl, truckContainer))

env.run(until=proc)
