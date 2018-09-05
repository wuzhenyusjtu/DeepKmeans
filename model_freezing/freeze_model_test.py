import tensorflow as tf
from tensorflow.python.framework import tensor_util
import argparse
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph, graph_def

import pickle
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="checkpoint/inception_v1_19_0.18/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph, graph_def = load_graph(args.frozen_model_filename)

    '''
    The following code is able to extract weights from the frozen model.
    And save the weights with the corresponding node name into a dictionary.
    The pickle is used here as a data format.
    '''
    var_dict = {}
    graph_nodes=[n for n in graph_def.node]
    wts = [n for n in graph_nodes if n.op == 'Const']
    for n in wts:
        #print("Name of the node - %s" % n.name)
        #print("Value - %s" % tensor_util.MakeNdarray(n.attr['value'].tensor))
        var_dict[n.name] = tensor_util.MakeNdarray(n.attr['value'].tensor)
    save_obj(var_dict, 'GoogLeNet_weights_dict')
    var_dict = load_obj('GoogLeNet_weights_dict')
    for key, value in var_dict.items():
        print(key, value)

    # We can verify that we can access the list of operations in the graph
    #for op in graph.get_operations():
    #    if any(x in op.name for x in ['weights', 'beta', 'gamma', 'moving_mean', 'moving_variance']):
    #        print(op.name)

    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/Image_Placeholder:0')
    y = graph.get_tensor_by_name('prefix/Predicted_Labels:0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:
    #    # Note: we don't nee to initialize/restore anything
    #    # There is no Variables in this graph, only hardcoded constants
        y_out = sess.run(y, feed_dict={
            x: np.random.randint(256, size=(48, 224, 224, 3))   # < 45
        })
        print(y_out)