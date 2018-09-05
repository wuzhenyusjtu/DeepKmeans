import tensorflow as tf
slim = tf.contrib.slim
from tf_flags import FLAGS

def inputs(dataset, image_preprocessing_fn, network_fn, num_epochs, batch_size):
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        shuffle=True,
        num_epochs=num_epochs,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    train_image_size = network_fn.default_image_size

    image = image_preprocessing_fn(image, train_image_size, train_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * batch_size)
    labels = slim.one_hot_encoding(
        labels, dataset.num_classes - FLAGS.labels_offset)
    print('Images shape is ', images.get_shape())
    print('Labels shape is ', labels.get_shape())
    return images, labels