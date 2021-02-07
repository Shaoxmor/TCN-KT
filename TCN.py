from model_function import *
import tensorflow as tf


def pca(x,dim):
    with tf.name_scope("PCA"):
        m,n= tf.to_float(x.get_shape()[0]),tf.to_int32(x.get_shape()[1])
        print("m=",m)
        print("n=",n)
        mean = tf.reduce_mean(x,axis=1)
        x_new = x - tf.reshape(mean,(-1,1))
        cov = tf.matmul(x_new,x_new,transpose_a=True)/(m - 1)
        e,v = tf.linalg.eigh(cov,name="eigh")
        e_index_sort = tf.math.top_k(e,sorted=True,k=dim)[1]
        v_new = tf.gather(v,indices=e_index_sort)
        pca = tf.matmul(x_new,v_new,transpose_b=True)
    return pca


class TCN(object):
    def __init__(self, batch_size, num_steps, num_skills):

        self.batch_size= batch_size
        self.num_steps = num_steps
        self.num_skills = num_skills

        self.input_data = tf.placeholder(tf.int32, [None, num_steps], name="input_data")
        self.next_id = tf.placeholder(tf.int32, [None, num_steps], name="next_id")
        self.target_id = tf.placeholder(tf.int32, [None], name="target_id")
        self.target_correctness = tf.placeholder(tf.float32, [None], name="target_correctness")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.same = tf.placeholder(tf.float32, [110,110], name="same")
        self.differ = tf.placeholder(tf.float32, [110,110], name="differ")

        end_d = 10
        d = tf.Variable(tf.contrib.layers.xavier_initializer()([num_skills-1,20]),dtype=tf.float32, trainable=True, name='w')
        trans_d = tf.transpose(d)
        temp = tf.matmul(d, trans_d)
        am = tf.sigmoid(temp, name='am')
        w = pca(am, end_d)

        up = tf.zeros([1, end_d])
        skill_w = tf.concat([up, w], axis=0, name='skill_w')
        next_skill = tf.nn.embedding_lookup(skill_w, self.next_id)
        zeros = tf.zeros([num_skills, end_d])
        t1 = tf.concat([skill_w, zeros], axis=-1)
        t2 = tf.concat([zeros, skill_w], axis=-1)
        input_w = tf.concat([t1, t2], axis=0)
        input_data = tf.nn.embedding_lookup(input_w, self.input_data)
        kernel_size = 6
        num_channels = [10]*6
        output_size = end_d
        outputs = TemporalCN(input_data, output_size, next_skill, num_channels , self.dropout_keep_prob, kernel_size)

        self.logits = tf.reduce_sum(outputs, axis=-1, name="logits")
        self.states = tf.sigmoid(self.logits, name="states")
        logits = tf.reshape(self.logits, [-1])
        selected_logits = tf.gather(logits, self.target_id)

        self.pred = tf.sigmoid(selected_logits, name="pred")
        loss1 =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=self.target_correctness))

        dia = tf.diag_part(am)
        temp = tf.ones([1, num_skills - 1])
        loss2 = tf.reduce_sum(tf.square(dia - temp))

        same_w = tf.reduce_sum(tf.square(am*self.same))
        differ_w = tf.reduce_sum(tf.square(am*self.differ))
        loss3 = differ_w-same_w

        self.loss = tf.reduce_mean(1000*loss1+10*loss2+0.01*loss3, name="losses")
