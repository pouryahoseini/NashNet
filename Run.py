from NashNet import NashNet

#Build a network instance
nash_net = NashNet('Config.cfg', configSection = "DEFAULT")

#Train the network
# nash_net.train()

#Evaluate the network
# nash_net.evaluate(num_to_print=None)

#Only print some examples
nash_net.printExamples(num_to_print=None)