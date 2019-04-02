import simpy

def car(env, name, bcs, driving_duration, charging_duration):
    #simulate driving time to bcs (battery charging station)
    yield env.timeout(driving_duration)
    print "%s arriving to bcs at %d" % (name, env.now)
    #request resource
    with bcs.request() as req:
        yield req #simulates waiting for resource
        #simulate charging
        print "%s starting to charge at %d" % (name, env.now)
        yield env.timeout(charging_duration)
        print "%s leaving bcs at %d" % (name, env.now)

env = simpy.Environment()
bcs = simpy.Resource(env, capacity=2)

#four cars in our simulation
for i in range(4):
    env.process(car(env, 'car %d' % i, bcs, i*2, 5))

env.run()
