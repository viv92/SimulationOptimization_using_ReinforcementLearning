import simpy

#depicting that priority resource allows for priority in waiting queue of the resource

def resource_user(env, name, res, wait, prio):
    yield env.timeout(wait)
    with res.request(priority=prio) as req:
        print "%s requesting resource at %d with priority %d" % (name, env.now, prio)
        yield req
        print "%s got resource at %s" % (name, env.now)
        yield env.timeout(3)
    print "%s done using resource at %d" % (name, env.now)

env = simpy.Environment()
resource = simpy.PriorityResource(env, capacity=1)
p1 = env.process(resource_user(env, 'p1', resource, wait=0, prio=0))
p2 = env.process(resource_user(env, 'p2', resource, wait=1, prio=0))
p3 = env.process(resource_user(env, 'p3', resource, wait=2, prio=-1)) #lower number = higher priority
# Although p3 requests the resource later than p2, it will get access to it earlier because its priority is higher
env.run()
