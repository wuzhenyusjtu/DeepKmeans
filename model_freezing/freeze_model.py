import os, argparse

import tensorflow as tf
import errno
from nets import GoogLeNet

os.environ["CUDA_VISIBLE_DEVICES"]=""


# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))

def get_tensors_ops_graph(sess):
    tvars = tf.trainable_variables()
    #tvars_vals = sess.run(tvars)
    #print('----------------------------Trainable Variables-----------------------------------------')
    #for var, val in zip(tvars, tvars_vals):
        #print(var.name, val)
    #for var in tvars:
    #    print(var.name)
    #print('----------------------------------------Operations-------------------------------------')
    #print([op.name for op in tf.get_default_graph().get_operations()])
    print('----------------------------------Nodes in the Graph---------------------------------------')
    print("\n".join([n.name for n in tf.get_default_graph().as_graph_def().node]))

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        network_fn = GoogLeNet.GoogLeNet(
            num_classes=1001,
            is_training=False)
        images_placeholder = tf.placeholder(tf.float32,
                                            shape=(None, network_fn.default_image_size,
                                                   network_fn.default_image_size, 3), name="Images_Placeholder")

        labels_placeholder = tf.placeholder(tf.int64, shape=(48, 1001), name="Labels_Placeholder")

        logits_op, _, _, _ = network_fn(images_placeholder)

        labels = tf.argmax(tf.nn.softmax(logits_op), axis=1, name="Predicted_Labels")

        right_count_top1_op = tf.reduce_sum(
                            tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_op), axis=1), tf.argmax(labels_placeholder, axis=1)), tf.int32), name="Right_Count_Top1")
        right_count_topk_op = tf.reduce_sum(
                            tf.cast(tf.nn.in_top_k(tf.nn.softmax(logits_op), tf.argmax(labels_placeholder, axis=1), 5), tf.int32), name="Right_Count_Topk")

        # We import the meta graph in the current default Graph
        #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        gvar_list = tf.global_variables()
        bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
        #print([var.name for var in bn_moving_vars])
        varlist = tf.trainable_variables()
        varlist += bn_moving_vars

        # We restore the weights
        saver = tf.train.Saver(varlist)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('######Session restored from trained model at {}!######'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_dir)

        get_tensors_ops_graph(sess)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="../checkpoint/saved_models/CR=1.5/inception_v1_37_0.32", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="Images_Placeholder,Labels_Placeholder,Predicted_Labels,Right_Count_Top1,Right_Count_Topk",
                        help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names)