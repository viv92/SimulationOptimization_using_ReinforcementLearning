import simpy

def example(env):
    val = yield env.timeout(5, value=42)
    print "now = %d, vaue = %d" % (env.now, val)

env = simpy.Environment()
env.process(example(env))
env.run()
