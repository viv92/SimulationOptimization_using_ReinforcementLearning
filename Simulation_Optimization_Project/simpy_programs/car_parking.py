import simpy

def car(env):
    while True:
        parking_duration = 5
        driving_duration = 2
        print "starting to park at %d" % env.now
        yield env.timeout(parking_duration)
        print "starting to drive at %d" % env.now
        yield env.timeout(driving_duration)


env = simpy.Environment()
env.process(car(env)) #env.process(generator())
env.run(until=15)
