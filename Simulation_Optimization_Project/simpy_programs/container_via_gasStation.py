import simpy

# depicting Container and its attributes:
# Container.level
# Container.capacity
# Container.put()
# Container.get()


class GasStation(object):
    def __init__(self, env):
        self.fuel_dispensor = simpy.Resource(env, capacity=1)
        self.gas_tank = simpy.Container(env, init=100, capacity=1000)
        self.mon_proc = env.process(self.monitor_tank(env)) #monitor_tank added as a process on  GasStation instantiated
    def monitor_tank(self, env):
        while True:
            print "monitored value=%d at %d" % (self.gas_tank.level, env.now)
            if self.gas_tank.level < 100:
                print "calling tanker at %d" % env.now
                yield env.process(tanker(env, self))
            print "monitoring delay starting at %d" % env.now
            yield env.timeout(10) #why? monitoring delay? -also required for other processes to get a chance to run. Else an endless while runs

def tanker(env, gas_station):
    yield env.timeout(8) #takes 10 to arrive
    print "tanker arrived at %d" % env.now
    amount = gas_station.gas_tank.capacity - gas_station.gas_tank.level
    yield gas_station.gas_tank.put(amount)

def car(name, env, gas_station):
    print "car %s arriving at %d" % (name, env.now)
    with gas_station.fuel_dispensor.request() as req:
        yield req
        print "car %s starts refueling at %d" % (name, env.now)
        yield gas_station.gas_tank.get(40)
        yield env.timeout(12) #refueling time
        print "car %s done refueling at %d" % (name, env.now)

def car_generator(env, gas_station): #to emulate car arrivals at different times
    for i in range(4):
        env.process(car(i, env, gas_station))
        yield env.timeout(5)

env = simpy.Environment()
gas_station = GasStation(env)
env.process(car_generator(env, gas_station))
env.run(100)
