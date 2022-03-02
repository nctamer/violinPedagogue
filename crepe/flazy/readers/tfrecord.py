import os
from tfrecord_lite import decode_example
from typing import Optional, List

import tensorflow.compat.v1 as tf

from .. import Dataset
from ..utils import close_iterator


def read_records(path: str, keys: List[str], options: tf.python_io.TFRecordOptions):
    if os.path.isdir(path):
        files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.tfrecord')]
    elif os.path.isfile(path):
        files = [path]
    else:
        raise ValueError('Could not open {}'.format(path))

    for file in files:
        iterator = tf.python_io.tf_record_iterator(file, options)
        try:
            for record in iterator:
                yield decode_example(record, keys or [])
        finally:
            close_iterator(iterator)


def tfrecord(*paths: str, keys: List[str]=None, compression: Optional[str]=None, **executor_config):
    options = None
    if compression and compression.lower() == 'gzip':
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    elif compression and compression.lower() == 'zlib':
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    return Dataset(paths).flatmap(lambda path: read_records(path, keys, options), **executor_config)
