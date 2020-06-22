# encoding=utf-8
import tensorflow as tf
import numpy as np
import os, random, csv, time, shutil
# Set log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 64

def get_images():
    test_path = 'test'
    files = os.listdir(test_path)
    files.sort()
    return files

def preprocess(file):
    image_graph = tf.Graph()
    with image_graph.as_default():
        image = tf.read_file(os.path.join('test', file))
        image_data = tf.image.decode_jpeg(image)
        if image_data.dtype != tf.float32:
            image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image_data = tf.image.central_crop(image_data, central_fraction=0.875)
        #######
        # print(tf.shape(image_data, name=None, out_type=tf.int32))
        # image_data = tf.squeeze(image_data, [0])
        # print(tf.shape(image_data, name=None, out_type=tf.int32))
        image_data = tf.subtract(image_data, 0.5)
        image_data = tf.multiply(image_data, 2.0)
        ######
        image_data = tf.image.resize_images(image_data, [299, 299])
        '''
        image_data = tf.expand_dims(image_data, 0)
        image_data = tf.image.resize_bilinear(image_data, [299, 299],
                                           align_corners=False)
        image_data = tf.squeeze(image_data, [0])
        image_data = tf.subtract(image_data, 0.5)
        image_data = tf.multiply(image_data, 2.0)
        '''
    with tf.Session(graph=image_graph) as image_sess:
        image_data = [image_sess.run(image_data)]
    return image_data
        
def infer(model, inference_sess, input_layer, output_layer):
    files = get_images()
    
    label_map_file = open('data_train/labels.txt')
    label_map = {}
    
    for line_number, label in enumerate(label_map_file.readlines()):
        label_map[line_number] = label[:-1]
        line_number += 1
    label_map_file.close()
    
    #import csv
    model_name = model.split('.')[0]
    if os.path.exists('ouput/{}.csv'.format(model_name)):
        os.remove('output/{}.csv'.format(model_name))
    csvfile = open('output/{}.csv'.format(model_name), 'a', encoding='utf-8')
    csvfile.write('ImageName,CategoryId\n')
    count = 1
    for file in files:
        time1 = time.time()
        image = preprocess(file)
        with tf.device('/gpu:0'):
            prediction = inference_sess.run(output_layer, feed_dict={input_layer: image})
        prediction = np.squeeze(prediction)
        prediction = np.argmax(prediction)
        prediction = label_map[prediction].split(':')[-1]
        time2 = time.time()
        
        print("#{} {}: {} | {}".format(count, file, prediction, str(time2-time1)))
        count += 1
        csvfile.write(str(file)+','+str(prediction)+'\n')
        '''
        if not os.path.exists('output/{}'.format(prediction)):
            os.mkdir('output/{}'.format(prediction))
        shutil.copyfile('data_raw/test/{}'.format(file), 'output/{}/{}'.format(prediction,file))
        '''
    csvfile.close()
    print("Finish.")
        
def main():
    model_dir = "./model"
    model = "nasnet_large.pb"
    model_path = os.path.join(model_dir, model)
    # print(model_path)
    model_graph = tf.Graph()
    with model_graph.as_default():
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            input_layer = model_graph.get_tensor_by_name("input:0")
            output_layer = model_graph.get_tensor_by_name('final_layer/predictions:0')
            print("Model Loading Success.")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inference_session = tf.Session(graph=model_graph, config=config)

    # Initialize session
    initializer = np.zeros([1, 299, 299, 3])
    inference_session.run(output_layer, feed_dict={input_layer: initializer})
    
    infer(
        model = model,
        inference_sess=inference_session,
        input_layer=input_layer,
        output_layer=output_layer
    )
    
def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    print(get_available_gpus())
    # print(get_images())
    main()
