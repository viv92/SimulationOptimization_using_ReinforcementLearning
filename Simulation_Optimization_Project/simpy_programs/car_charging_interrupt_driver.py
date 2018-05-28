import simpy

class Car(object):
    def __init__(self, env):
        self.env = env
        self.action = self.env.process(self.run())
    def run(self):
        while True:
            try:
                print "start charging at %d" % self.env.now
                yield self.env.process(self.charge())
            except simpy.Interrupt:
                print "driver interrupts charging at %d. Hope charge is enough" % self.env.now
            print "start driving at %d" % self.env.now
            driving_duration = 2
            yield self.env.timeout(driving_duration)
    def charge(self):
        charging_duration = 5
        yield self.env.timeout(charging_duration)

def driver(env, car):
    #interrupts car's run process after 3 seconds => interrupts charging
    yield env.timeout(3)
    car.action.interrupt() #requires access to the return value of the process to be interrupted

env = simpy.Environment()
car = Car(env)
env.process(driver(env, car))
env.run(until=15)
