import depthai as dai
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Create Script node
script = pipeline.create(dai.node.Script)

# Create NeuralNetwork node
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("testBuff_simplified_openvino_2022.1_6shave.blob")  # Load NN model

# Link Script node to NN node
script.outputs['nnInp'].link(nn.input)

# Create an XLinkOut to get the NN output
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn_out")
nn.out.link(xout_nn.input)

# Define the script node's function
script.setScript("""
# Create NNData message
nn_data = NNData(64) # Not sure if size is 32? Is this the size in bits or bytes? How to calculate this?
nn_data.setLayer("fp16", [21.0])
node.io['nnInp'].send(nn_data)
""")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    q_nn_out = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)

    while True:
        in_nn = q_nn_out.get()
        if in_nn is not None:
            print("Received NN output:", np.array(in_nn.getFirstLayerFp16()))  # Get the NN output
