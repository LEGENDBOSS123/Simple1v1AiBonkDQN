import * as tf from '@tensorflow/tfjs';
tf.setBackend('webgl');
top.tf = tf;
export { tf };