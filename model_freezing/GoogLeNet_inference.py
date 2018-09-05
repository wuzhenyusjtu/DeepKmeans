import tensorflow as tf
import argparse
import numpy as np
import os
from datasets import imagenet

from preprocessing import preprocessing_factory
from nets import GoogLeNet

slim = tf.contrib.slim

# Not using any GPU
os.environ["CUDA_VISIBLE_DEVICES"]=""


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def inputs(dataset, image_preprocessing_fn, num_epochs, batch_size):
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=4,
        shuffle=True,
        num_epochs=num_epochs,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [image, label] = provider.get(['image', 'label'])

    train_image_size = 224

    image = image_preprocessing_fn(image, train_image_size, train_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size)
    labels = slim.one_hot_encoding(
        labels, dataset.num_classes)
    print('Images shape is ', images.get_shape())
    print('Labels shape is ', labels.get_shape())
    return images, labels

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="frozen_model_cr=3.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

    # We access the input and output nodes in the graph
    images_placeholder = graph.get_tensor_by_name('prefix/Images_Placeholder:0')
    labels_placeholder = graph.get_tensor_by_name('prefix/Labels_Placeholder:0')

    predicted_labels = graph.get_tensor_by_name('prefix/Predicted_Labels:0')

    right_count_top1_op = graph.get_tensor_by_name('prefix/Right_Count_Top1:0')
    right_count_topk_op = graph.get_tensor_by_name('prefix/Right_Count_Topk:0')

    dataset = imagenet.get_split(
        'validation', '../datasets/imagenet-data/tfrecords')

    with tf.Session(graph=graph) as sess:
        images_op, labels_op = inputs(dataset=dataset, image_preprocessing_fn=preprocessing_factory.get_preprocessing(is_training=False),
                                                     num_epochs=1, batch_size=48)

        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        total_v = 0.0
        test_correct_num_top1 = 0.0
        test_correct_num_topk = 0.0

        from tqdm import tqdm

        pbar = tqdm(total=dataset.num_samples // (48),)

        try:
            while not coord.should_stop():
                pbar.update(1)
                images, labels = sess.run([images_op, labels_op])

                right_count_top1, right_count_topk = sess.run(
                                                [right_count_top1_op, right_count_topk_op],
                                               feed_dict={images_placeholder: images,
                                                          labels_placeholder: labels,})


                test_correct_num_top1 += right_count_top1
                test_correct_num_topk += right_count_topk
                total_v += labels.shape[0]

        except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
        finally:
            coord.request_stop()
        print('Test acc top1:', test_correct_num_top1 / total_v,
                  'Test_correct_num top1:', test_correct_num_top1,
                  'Total_v:', total_v)
        print('Test acc topk:', test_correct_num_topk / total_v,
                  'Test_correct_num topk:', test_correct_num_topk,
                  'Total_v:', total_v)



    # # We launch a Session
    # with tf.Session(graph=graph) as sess:
    #     # Feed the network with randomly generately image in [0,255] scale
    #     # 48: batch size; 224: imagenet default image size; 3: rgb channels
    #     y_out = sess.run(y, feed_dict={
    #         x: np.random.randint(256, size=(48, 224, 224, 3))   # < 45
    #     })
    #     print(y_out)
