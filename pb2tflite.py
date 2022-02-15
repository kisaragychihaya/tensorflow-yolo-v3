import tensorflow as tf
print(tf.__version__)
if str(tf.__version__).startswith( '1' ):
    print("Need TF 2.X")
    exit(1)
import tempfile
import os
import glob
import tensorflow.compat.v1 as tf
gf = tf.GraphDef()   
m_file = open('/content/gdrive/MyDrive/frozen_darknet_yolov3_model.pb','rb')
gf.ParseFromString(m_file.read())

with open('somefile.txt', 'a') as the_file:
    for n in gf.node:
        the_file.write(n.name+'\n')

file = open('somefile.txt','r')
data = file.readlines()
output = data[len(data)-1]
print("Output array = ", output)

file.seek ( 0 )
input=file.readline()
print("Input array = ", input)
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='frozen_darknet_yolov3_model.pb', 
    input_arrays = ['inputs'],   # Here, 'inputs' is the value of input array from Step 7b
    output_arrays = ['output_boxes'], # Here, 'output_boxes' is the value of output array from Step 7b
    input_shapes={'inputs': [1, 416, 416, 3]} # Here, 'inputs' is the value of input array from Step 7b
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert to TFLite Model
tflite_model = converter.convert()

_, dynamic_tflite_path = tempfile.mkstemp('.tflite')
tflite_model_size = open(dynamic_tflite_path, 'wb').write(tflite_model)
tf_model_size = os.path.getsize('frozen_darknet_yolov3_model.pb')
print('TensorFlow Model is  {} bytes'.format(tf_model_size))
print('TFLite Model is      {} bytes'.format(tflite_model_size))
print('Post training dynamic range quantization saves {} bytes'.format(tf_model_size-tflite_model_size))
print("TF Lite model Save to:%s"%dynamic_tflite_path)
