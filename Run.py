import sys
from NashNet import NashNet

# Decide on the config file address
if len(sys.argv) == 1:
    config_file_address = './Config/Config.cfg'
else:
    config_file_address = sys.argv[1]

# Build a network instance
nash_net = NashNet(config_file_address, configSection="DEFAULT")

# Train the network
nash_net.train()

# Evaluate the network
nash_net.evaluate(num_to_print=None)

# Only print some examples
nash_net.printExamples(num_to_print=None)

