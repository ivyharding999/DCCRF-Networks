#!/usr/bin/env python3
###[Finished in 7409.5s]   without BN
###[Finished in 7575.9s]   with  BN
###[Finished in 6174.9s]   eigen2_crfrnn_net running
###[Finished in 6270.2s]      alex_crfrnn 
import tensorflow as tf
from datetime import datetime
import model_scale3 as m
import net_input as feed_in
import h5py
import time
import matplotlib.pyplot as plt
import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_step', 80000, 'Number of max step')
tf.app.flags.DEFINE_integer('batch_size', 4 , 'batch size')
tf.app.flags.DEFINE_float('lr', 1e-4, 'Learing rate')


def write_hdf5(file_name, losses):
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('losses', data=losses)

def train():
    time_start=time.time()
    data = feed_in.read_file('train_multidim_data.h5')
    with tf.Graph().as_default():
        x0 = tf.placeholder(tf.float32, [FLAGS.batch_size, 40, 144, 1])
        images = tf.placeholder(tf.float32, [FLAGS.batch_size, 160, 576, 1])
        labels = tf.placeholder(tf.int64, [FLAGS.batch_size, 40, 144, 1])
        global_step = tf.Variable(1, trainable=False)       
        pre= m.inference_1(images)
        loss = m.loss(pre, labels, 20, 25 , 28, 3)
        train_op = m.train(loss, global_step)
        saver = tf.train.Saver()
        losses_save=[]
        losses = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())   
            format_str = "%s    Iteration =  %d   loss = %0.3f"
            for step in range(FLAGS.max_step + 1):
                batch = data.next_batch(FLAGS.batch_size)
                loss_val, result= sess.run([loss, train_op], feed_dict={x0: batch[1], images: batch[3], labels: batch[4]})
        
                losses.append(loss_val)
                
                if step % 20 == 0:
                    print('loss:',format_str%(datetime.now(), step, loss_val)) 
                    losses_save.append(loss_val) 
                

                if step % 40000 == 0:
                    checkpoint = 'scale3-80000_new/' + 'model.ckpt'
                    saver.save(sess, checkpoint, global_step=step) 

                if step % 80000 == 0:
                    write_hdf5('cost_scale3.h5', losses_save)
                    write_file=open('loss_scale3_new.pkl','wb')  
                    pickle.dump(losses_save,write_file)  # 将数据存入pickle
                    write_file1=open('loss_scale3_80000_new.pkl','wb')  
                    pickle.dump(losses,write_file1)  # 将数据存入pickle
                    write_file1.close() 
                    write_file.close()

            
            plt.plot(losses_save)
            plt.plot(losses)
            plt.title('train_loss')
            plt.xlabel(u'iter')
            plt.ylabel(u'loss')
            plt.savefig("loss_scale3_new.png")
            plt.show()

    time_end=time.time()
    print('spend time:',time_end-time_start,'s')
 

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
