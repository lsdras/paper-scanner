import tensorflow as tf
import os.path
import warnings
from distutils.version import LooseVersion
from glob import glob
import helper
import sys
import time
from time import localtime

now = time.strftime("%m%d%H%M%S",localtime())


# --------------------------
# MOBILENET
# --------------------------


import tensorflow as tf
import tensorflow.nn as nn

import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    tchannel = inputs.shape[-1] * t

    x = slim.conv2d(inputs, tchannel, 1, 1, padding='same', activation_fn=nn.relu6, normalizer_fn=slim.batch_norm)

    x = slim.convolution2d_in_plane(x, kernel, stride=s, padding='SAME', activation_fn=nn.relu6, normalizer_fn=slim.batch_norm)

    x = slim.conv2d(x, filters, 1, 1, padding='same', activation_fn=None, normalizer_fn=slim.batch_norm)

    if r:
        x = tf.add(x, inputs)
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def MobileNetv2():
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """

    inputs = tf.placeholder(tf.float32,[None,None,None,3])
    keep_prob = tf.placeholder(tf.float32)
    x = slim.conv2d(inputs, 32, (3,3), stride=2, activation_fn=nn.relu6, normalizer_fn=slim.batch_norm)

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    layer1 = x #fcn8
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    layer2 = x #fcn16
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)

    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
    x = slim.dropout(x,keep_prob)
    x = slim.conv2d(x, 1280, 1, stride=1, activation_fn=nn.relu6, normalizer_fn=slim.batch_norm)
    x = slim.dropout(x,keep_prob)
    layer3 = x #fcn32

    return inputs, keep_prob, layer1, layer2, layer3


# --------------------------
# FUNCTIONS
# --------------------------


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
                                      kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
                                       kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
                                       kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")
    
    return fcn11


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))
    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")
    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")        
    return logits, train_op, loss_op
'''

# Direct IOU optimizer, learning rate =0.0001~0.0005(0.001 is too big)    
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))
    just_correct = tf.matmul(correct_label_reshaped,tf.constant([[0],[1]],tf.float32))
    proposal = tf.matmul(tf.nn.softmax(logits),tf.constant([[0],[1]],tf.float32))    
    inter = tf.reduce_sum(tf.matmul(tf.transpose(proposal),just_correct))
    union = tf.add(tf.reduce_sum(tf.add(just_correct, proposal)),-inter)
    loss_op = -tf.log(tf.div(inter,union))
    
    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")
    

    return logits, train_op, loss_op
'''

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    keep_prob_value = 0.5
    learning_rate_value = 0.001
    global_step = 0
    for epoch in range(epochs):
        print('epoch:',epoch+1,'/',epochs)
        # Create function to get batches
        total_loss = 0
        total_iou = 0
        iter = 0
        print('Total:',NUM_OF_IMAGES,'/ batch:',BATCH_SIZE)
        for X_batch, gt_batch in get_batches_fn(batch_size):
            iter += BATCH_SIZE
            print('$',end='',flush=True)
            loss, _ = sess.run([cross_entropy_loss, train_op],
                               feed_dict={input_image: X_batch, correct_label: gt_batch,
                                          keep_prob: keep_prob_value, learning_rate: learning_rate_value})
            #loss_summ = tf.summary.scalar("loss", loss)
            #merge = tf.summary.merge_all()
            #summary = sess.run(merge)
            #writer.add_summary(summary, global_step)
            #global_step += 1 

            total_loss += loss 
        print('\n')
        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()


def run():

    # A function to get batches
    get_batches_fn = helper.gen_batch_function(training_dir, IMAGE_SHAPE)    
    
    with tf.Session() as session:
        
        # Returns the three layers, keep probability and input layer from the vgg architecture
        image_input, keep_prob, layer3, layer4, layer7 = MobileNetv2()


        # The resulting network architecture from adding a decoder on top of the given vgg model
        model_output = layers(layer3, layer4, layer7, num_classes)
        
        saver = tf.train.Saver()
        
        # Returns the output logits, training operation and cost operation to be used
        # - logits: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)

        #writer = tf.summary.FileWriter('./logs/', session.graph)
        
        # Initialize all variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        print("Model build successful, starting training")
        
        # Train the neural network
        train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn,
                 train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)
                      
        # Run the model with the test images and save each painted output image (roads painted green)
        #helper.save_inference_samples(runs_dir, training_dir, session, IMAGE_SHAPE, logits, keep_prob, image_input,now, mask_only=False)

        os.mkdir('./logs/%s' % now)
        save_path = saver.save(session, "./logs/{0}/{1}".format(now,now))
        print("Model saved in path: %s" % save_path)
        
        helper.save_inference_samples(runs_dir, training_dir, session, IMAGE_SHAPE, logits, keep_prob, image_input, now, mask_only=False)

        

        print("All done!")
        
def detect_run(weight_path):
    pass


# --------------------------
# MAIN
# --------------------------
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='note segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights",
                        help="Path to weights")
    parser.add_argument('--datadir', required=False,
                        metavar="path to dataset",
                        help="Path to dataset")
    parser.add_argument('--imgsize', required=False,
                        metavar="(h,w), h=w",type=int,
                        help="(h,w),h=w, input int")
    args = parser.parse_args()


    # --------------------------
    # USER-SPECIFIED DATA
    # --------------------------

    # Specify these directory paths

    data_dir = './data'
    runs_dir = './runs'
    training_dir = args.datadir

    # Tune these parameters

    num_classes = 2
    IMAGE_SHAPE = (args.imgsize,args.imgsize)
    EPOCHS = 40
    BATCH_SIZE = 2
    NUM_OF_IMAGES = len(glob(os.path.join(training_dir, 'images','*.png')))    

    # --------------------------
    # PLACEHOLDER TENSORS
    # --------------------------

    correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], num_classes])
    #label_weight_mask = tf.constant(tf.float32, )
    learning_rate = tf.placeholder(tf.float32)


  # Train or evaluate
    if args.command == "train":
        run()
    elif args.command == "detect":
        detect_run(args.weights)
    else:
        print("'{}' is not recognized. "
            "Use 'train' or 'detect'".format(args.command))
