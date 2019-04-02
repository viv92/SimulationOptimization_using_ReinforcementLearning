import simpy
import numpy as np
import scipy
import matplotlib.pyplot as plt
import mcerp

np.random.seed(0)

num_of_load = 0
num_of_haul = 0
num_of_dump = 0
num_of_return = 0

def earthmov_operation(env, truckContainer, soilInStkPl, soilInDump, whlLdr):
    #global cost
    #cost = 0.0
    for i in range(5): # 5 trucks
        env.process(operate_truck(env, 'truck %d' % i, truckContainer, soilInStkPl, soilInDump, whlLdr))

    #while True:
    #    cost += pass
    yield env.timeout(1)

def operate_truck(env, name, truckContainer, soilInStkPl, soilInDump, whlLdr):
    global num_of_load
    global num_of_haul
    global num_of_dump
    global num_of_return
    while True:
        with whlLdr.request() as req:
            yield req
            yield soilInStkPl.get(15)
            yield truckContainer.get(1)
            #print "%s goes out of container at %r for loading" % (name, env.now)
            yield env.timeout(load_time())
        num_of_load += 1
        #print "%s done loading. goes for haul at %r" % (name, env.now)
        yield env.timeout(haul_time())
        num_of_haul += 1
        #print "%s done hauling. goes for dump at %r" % (name, env.now)
        yield env.timeout(dump_time())
        yield soilInDump.put(15)
        num_of_dump += 1
        #print "%s done dumping. starts returning at %r" % (name, env.now)
        yield env.timeout(return_time())
        yield truckContainer.put(1)
        num_of_return += 1
        #print "%s returns to container at %r" % (name, env.now)

def monitor_soilInStkPl(env, truckContainer, soilInStkPl):
    global num_of_load
    global num_of_haul
    global num_of_dump
    global num_of_return
    round = 0
    while True:
        round += 1
        print "monitoring round=%d\t soilInStkPl=%d\t truckContainer=%d" % (round, soilInStkPl.level, truckContainer.level)
        if (soilInStkPl.level < 15) and (truckContainer.level == 5):
            print "Soil in Stock Pile < 15"
            print """
            num_of_load = %d \n
            num_of_haul = %d \n
            num_of_dump = %d \n
            num_of_return = %d \n
            """ % (num_of_load, num_of_haul, num_of_dump, num_of_return)
            return
        yield env.timeout(1)

def load_time():
    return np.random.uniform(1.3, 1.8)

def haul_time():
    return mcerp.PERT(4, 5.5, 6)

def dump_time():
    return 0.5

def return_time():
    return mcerp.PERT(3, 4, 5)

env = simpy.Environment()
truckContainer = simpy.Container(env, init=5, capacity=6)
soilInStkPl = simpy.Container(env, init=1000, capacity=1000)
soilInDump = simpy.Container(env, init=0, capacity=1000)
whlLdr = simpy.Resource(env, capacity=1)
env.process(earthmov_operation(env, truckContainer, soilInStkPl, soilInDump, whlLdr))
proc = env.process(monitor_soilInStkPl(env, truckContainer, soilInStkPl))
env.run(until=proc)
