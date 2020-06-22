from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

GRAPH_PB_PATH = 'inference/inception_resnet_v2.pb' #path to your .pb file
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
        # Note: one of the following two lines work if required libraries are available
        #text_format.Merge(f.read(), graph_def)
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        for i,n in enumerate(graph_def.node):
            '''
            if i == 1 or i == len(graph_def.node)-1:
                print("Name of the node - %s/n" % n.name)
            '''
            print("Name of the node - %s/n" % n.name)