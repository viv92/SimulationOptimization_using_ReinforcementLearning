import simpy

# following resource attributes are used / depicted:
# Resource.request()
# Resource.count
# Resource.capacity
# Resource.users
# Resource.queue

def print_stats(res):
    print "%d slots of %d are allocated" % (res.count, res.capacity)
    print "current users: ", res.users
    print "queued events: ", res.queue

def user(res):
    print_stats(res)
    with res.request() as req: #generate a request event
        yield req #wait for access
        print_stats(res) #do something on access
    print_stats(res) #resource released automatically

env = simpy.Environment()
resource = simpy.Resource(env, capacity=1) #instantiate resource
procs = [env.process(user(resource)), env.process(user(resource))] #instantiate two users
env.run()
