import simpy

class Car(object):
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.action = env.process(self.run()) #always call run process on each instantiation
    def run(self):
        while True:
            print "%s start charging at %d" % (self.name, self.env.now)
            yield self.env.process(self.charge())
            print "%s start driving at %d" % (self.name, self.env.now)
            driving_duration = 2
            yield self.env.timeout(driving_duration)
    def charge(self):
        charging_duration = 5
        yield self.env.timeout(charging_duration)

env = simpy.Environment()
car1 = Car(env, 'car1')
car2 = Car(env, 'car2')
env.run(until=15)
