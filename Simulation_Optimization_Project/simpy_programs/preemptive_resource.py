import simpy

#depicting that preemptive resource allows a higher priority user to kick out a
#lower priority user from using the resource

#also depicts:
#Interrupt.cause.by - returns the interrupt causing process
#Interrupt.cause.usage_since - returns the time at which the interrupted process had started

def resource_user(env, name, res, wait, prio):
    yield env.timeout(wait)
    with res.request(priority=prio) as req:
        print "%s requesting resource at %d with priority %d" % (name, env.now, prio)
        yield req
        print "%s got resource at %s" % (name, env.now)
        try:
            yield env.timeout(3)
        except simpy.Interrupt as interrupt:
            by = interrupt.cause.by
            usage = env.now - interrupt.cause.usage_since
            print "%s got preempted by %s at %d after using the resource for %d" % (
            name, by, env.now, usage)

env = simpy.Environment()
resource = simpy.PreemptiveResource(env, capacity=1)
p1 = env.process(resource_user(env, 'p1', resource, wait=0, prio=0))
p2 = env.process(resource_user(env, 'p2', resource, wait=1, prio=0))
p3 = env.process(resource_user(env, 'p3', resource, wait=2, prio=-1)) #lower number = higher priority
#p3 preempts p1
env.run()
