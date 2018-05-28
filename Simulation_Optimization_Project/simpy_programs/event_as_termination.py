import simpy

def fn(env):
    yield env.timeout(7)
    print 'Monty Python is over'

env = simpy.Environment()
proc = env.process(fn(env)) #event (process in this case) used for termination
env.run(until=proc)
