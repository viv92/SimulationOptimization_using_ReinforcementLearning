import numpy as np
import simpy
import matplotlib.pyplot as plt

#initial seed for np.random
np.random.seed(0)

#initalize environment
env = simpy.Environment()

#factory run process
def factory_run(env, repairers, spares):
    #initialise cost (global variable)
    global cost
    cost = 0.0
    #add machine run process for 50 machines - to be done once
    for i in range(50):
        env.process(operate_machine(env, repairers, spares))

    while True:
        #update daily cost incurred due to repairers and spares
        cost += 3.75*8*repairers.capacity + 30*spares.capacity #level?
        yield env.timeout(8) #wait a day for next update

def operate_machine(env, repairers, spares):
    global cost
    while True:
        yield env.timeout(time_to_fail()) #wait for a machine to break
        t_broken = env.now #note time at which a machine broke
        print "a machine broke at %.2f" % t_broken
        #launch repair process
        env.process(repair_machine(env, repairers, spares))
        #launch replacement process
        yield spares.get(1) #wait for replacement process to complete
        t_replaced = env.now #note time at which machine replaced
        print "the machine replaced at %.2f" % t_replaced
        cost += 20*(t_replaced - t_broken) #update opportunity cost

def repair_machine(env, repairers, spares):
    with repairers.request() as req:
        yield req #wait for access to a repairer
        yield env.timeout(time_to_repair()) #wait for repair to complete
        yield spares.put(1) #replenish spares with repaired machine
    print "repair complete at %.2f" % env.now

def time_to_repair():
    return np.random.uniform(4, 10)

def time_to_fail():
    return np.random.uniform(132, 182) #uniform distribution

#monitor process
obs_time = []
obs_cost = []
obs_spares = []
def observe(env, spares):
    while True:
        obs_time.append(env.now)
        obs_cost.append(cost)
        obs_spares.append(spares.level)
        yield env.timeout(1.0) #monitor process runs hourly

#repairers as resource
repairers = simpy.Resource(env, capacity=3)
#spares as container
spares = simpy.Container(env, init=15, capacity=15)
#add factory_run process
env.process(factory_run(env, repairers, spares))
#add monitor process
env.process(observe(env, spares))
#run environment for a year (8 hours a day, 5 days a week, 52 weeks)
env.run(until=8*5*52)

print "final cost: %.2f" % obs_cost[-1]

#plot
plt.figure()
plt.step(obs_time, obs_spares, where='post')
plt.xlabel('time (hours)')
plt.ylabel('spares level')
plt.show()

plt.figure()
plt.step(obs_time, obs_cost, where='post')
plt.xlabel('time (hours)')
plt.ylabel('cost')
plt.show()
