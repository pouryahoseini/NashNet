import sys
from NashNet import NashNet

# Build a network instance
nash_net = NashNet(sys.argv[1], configSection="DEFAULT")

# Train the network
nash_net.train()

# Evaluate the network
nash_net.evaluate(num_to_print=None)

# Only print some examples
nash_net.printExamples(num_to_print=None)

