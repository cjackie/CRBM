import tensorflow as tf
import numpy as np
import scipy.stats as stats
import logging

class CRBM():
    '''
    CRBM based on a paper 'Unsupervised feature learning for audio classification using convolutional
    deep belief networks' by Honglak Lee, Yan Largman, Peter Pham, and Andrew Y, Ng.
    '''
    def __init__(self, filter_shape, visible_shape, k, params_id, stddev=1.0, binary=False):
        '''

        :param filter_shape: tuple of integer (f_h, f_w). shape of filter. f_h, f_w are height and width respectively.
        :param visible_shape: tuple of integer (v_h, v_w). shape of visibles. v_h, v_w are height and width respectively.
        :param k: integer. also called group, number of filters.
        :param params_id: str. unique identifier for parameters.
        :param stddev: float. standard deviation during parameters initialization
        :param binary: bool. if visible units are binary or not. False means they are real values.
        '''
        assert filter_shape[0] <= visible_shape[0]
        assert filter_shape[1] <= visible_shape[1]
        if binary:
            raise Exception('not implemented yet')

        graph = tf.Graph()
        with graph.as_default() as g:
            with tf.variable_scope(params_id) as crbm_scope:
                self.w = tf.get_variable('weights', shape=(k,) + filter_shape,
                                         initializer=tf.random_normal_initializer(mean=0.0, stddev=stddev))
                self.w_r = self.w[:,:,::-1]
                self.hb = tf.get_variable('hidden_biases', shape=(k,),
                             initializer=tf.random_normal_initializer(mean=0.0, stddev=stddev))
                self.vb = tf.get_variable('visible_biases', shape=(1,),
                             initializer=tf.random_normal_initializer(mean=0.0, stddev=stddev))

            self.sess = tf.Session(graph=graph)
            # initialize parameters
            self.sess.run(tf.global_variables_initializer())

        self.hidden_shape = (visible_shape[0]-filter_shape[0]+1, visible_shape[1]-filter_shape[1]+1)
        self.filter_shape = filter_shape
        self.visible_shape = visible_shape
        self.k = k
        self.params_id = params_id
        self.binary = binary
        self.graph = graph

    def generate_hidden_units_probabilities(self, visible):
        '''
        generate probabilities of hidden units being 1, given visible units
        :param v: numpy array of shape (b,) + self.visible_shape. b is batch size.
        :return: numpy array of shape (b,) + self.k + self.hidden_shape. b is batch size. each element is between [0,1]
        '''
        assert visible.shape[1:] == self.visible_shape

        w = self.w
        hb = self.hb
        sess = self.sess
        graph = self.graph

        with graph.as_default() as g:
            visible = tf.convert_to_tensor(visible)

            # fit data.
            visible = tf.expand_dims(visible, axis=-1)
            w = tf.expand_dims(w, axis=0)
            w = tf.transpose(w, perm=[2,3,0,1]) # (h, w, 1, k)

            tf_convolution = tf.nn.conv2d(visible, w, [1,1,1,1], 'VALID') + hb
            tf_convolution = tf.transpose(tf_convolution, perm=[0,3,1,2]) # (b, k, h, w)
            convoluted = sess.run(tf_convolution)

        probabilities = 1.0/(1+np.power(np.e, -convoluted))
        return probabilities

    def generate_hidden_units(self, visible):
        '''
        generate hidden units (1s and 0s), given visible units.
        :param v: numpy array of shape (b,) + self.visible_shape. visible units. b is batch size.
        :return: numpy array of shape (b,) + self.k + self.hidden_shape. b is batch size. each element is between [0,1]
        '''
        assert visible.shape[1:] == self.visible_shape

        hidden_shape = self.hidden_shape
        k = self.k

        batch_size = visible.shape[0]

        probabilities = self.generate_hidden_units_probabilities(visible)
        # efficient sampling
        random_0to1 = np.random.rand(batch_size,k,hidden_shape[0],hidden_shape[1])
        eval_func = np.vectorize(lambda d: 1 if d > 0 else 0)
        hidden_units = eval_func(probabilities - random_0to1)
        return hidden_units

    def generate_visible_units_expectations(self, hidden):
        '''
        generate expectations(mean) of visible units.
        :param hidden: numpy array of shape (b,) + self.k + self.hidden_shape.
        :return: numpy array of shape (b,) + self.visible_shape.
        '''
        assert hidden.shape[1:] == (self.k,) + self.hidden_shape
        if self.binary:
            raise Exception('not implemented yet')

        filter_shape = self.filter_shape
        w_r = self.w_r
        vb = self.vb
        sess = self.sess
        graph = self.graph

        filter_height = filter_shape[0]
        filter_width = filter_shape[1]

        with graph.as_default() as g:
            hidden = tf.convert_to_tensor(hidden, dtype=tf.float32)

            w_r = tf.expand_dims(w_r, axis=0)
            w_r = tf.transpose(w_r, perm=[2,3,1,0]) # (h, w, k, 1)
            padding = tf.convert_to_tensor([[0,0],[filter_height-1,filter_height-1],
                                            [filter_width-1,filter_width-1],[0,0]])
            hidden = tf.transpose(hidden, perm=[0,2,3,1])
            hidden = tf.pad(hidden, padding) # (b, h, w, k)
            tf_expectation = tf.nn.conv2d(hidden, w_r, [1,1,1,1], 'VALID') + vb  # (b, h, w, 1)
            expectations = sess.run(tf.squeeze(tf_expectation, axis=[-1]))

        return expectations

    def generate_visible_units(self, hidden, sigma=1):
        '''
        generate visible units.
        :param hidden: numpy array of shape (b,) + self.k + self.hidden_shape.
        :param sigma: float. variance of generating visible units.
        :return: numpy array of shape (b,) + self.visible_shape.
        '''
        assert hidden.shape[1:] == (self.k,) + self.hidden_shape
        if self.binary:
            raise Exception('not implemented yet')

        expectations = self.generate_visible_units_expectations(hidden)
        sample_func = np.vectorize(lambda ep : stats.norm.rvs(ep,sigma))
        visible = sample_func(expectations)
        return visible

    class Trainer:
        '''
        Trainer to CRBM using constrastive divergence(CD).
        '''
        def __init__(self, crbm, summary_enabled=False, summary_dir=None, summary_flush_secs=60, init_step=0):
            '''
            an instance of CRBM to be trained.
            :param crbm: CRBM.
            :param summary_enabled: bool. enable summary
            :param summary_dir: str. when @summary_enable, this is where summary will be saved.
            :param init_step: int. step to begin. for resuming from previous training, set it to the previous step + 1.
            '''
            if summary_enabled:
                assert summary_dir != None

            filter_shape = crbm.filter_shape
            visible_shape = crbm.visible_shape
            hidden_shape = crbm.hidden_shape
            k = crbm.k
            w = crbm.w
            hb = crbm.hb
            vb = crbm.vb
            graph = crbm.graph
            tf_dtype = tf.float32

            summaries, summary_file = None, None
            with graph.as_default() as g:
                w = tf.expand_dims(w, axis=0)
                w = tf.transpose(w, perm=[2,3,0,1]) # (h, w, 1, k)

                hidden_in = tf.placeholder(tf_dtype, shape=[None, k, hidden_shape[0],hidden_shape[1]])
                visible_in = tf.placeholder(tf_dtype, shape=[None, visible_shape[0], visible_shape[1]])

                hidden_in_fitted = tf.transpose(hidden_in, perm=[0,2,3,1]) # (b, h, w, k)
                visible_in_fitted = tf.expand_dims(visible_in, axis=-1) # (b, h, w, 1)

                convolution = tf.nn.conv2d(visible_in_fitted, w, [1,1,1,1], 'VALID')
                energy = -tf.reduce_sum(hidden_in_fitted*convolution, axis=[1,2,3]) \
                                - tf.reduce_sum(hb*tf.reduce_sum(hidden_in_fitted, axis=[1,2]), axis=[1]) \
                                - vb*tf.reduce_sum(visible_in_fitted, axis=[1,2,3])

                hidden_rec_in = tf.placeholder(tf_dtype, shape=[None, k, hidden_shape[0],hidden_shape[1]])
                visible_rec_in = tf.placeholder(tf_dtype, shape=[None, visible_shape[0], visible_shape[1]])

                hidden_rec_in_fitted = tf.transpose(hidden_rec_in, perm=[0,2,3,1])
                visible_rec_in_fitted = tf.expand_dims(visible_rec_in, axis=-1)
                convolution_rec = tf.nn.conv2d(visible_rec_in_fitted, w, [1,1,1,1], 'VALID')
                energy_rec = -tf.reduce_sum(hidden_rec_in_fitted*convolution_rec, axis=[1,2,3]) \
                                - tf.reduce_sum(hb*tf.reduce_sum(hidden_rec_in_fitted, axis=[1,2]), axis=[1]) \
                                - vb*tf.reduce_sum(visible_rec_in_fitted, axis=[1,2,3])

                probabilities = tf.reduce_mean(tf.nn.sigmoid(convolution + hb)) # it is also the regularization term.
                loss = tf.reduce_mean(energy - energy_rec)
                learning_rate_in = tf.placeholder(tf_dtype)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate_in)
                energy_mean = tf.reduce_mean(energy)
                minimize_energy = optimizer.minimize(energy_mean)
                energy_rec_mean = tf.reduce_mean(energy_rec)
                maximize_energy_rec = optimizer.minimize(-energy_rec_mean)

                if summary_enabled:
                    summary_file = tf.summary.FileWriter(summary_dir, graph, flush_secs=summary_flush_secs)
                    tf.summary.scalar('loss', loss)
                    tf.summary.scalar('probability', probabilities)
                    tf.summary.scalar('real_energy', energy_mean)
                    tf.summary.scalar('reconstructed_energy', energy_rec_mean)
                    w_gradient = tf.gradients(loss, w)
                    tf.summary.histogram('weights_gradient', w_gradient)
                    hb_gradient = tf.gradients(loss, hb)
                    tf.summary.histogram('hidden_biases_gradient', hb_gradient)
                    vb_gradient = tf.gradients(loss, vb)
                    tf.summary.histogram('visible_biases_gradient', vb_gradient)
                    summaries = tf.summary.merge_all()

            self.crbm = crbm
            self.probabilities = probabilities
            self.loss = loss
            self.energy_mean = energy_mean
            self.energy_rec_mean = energy_rec_mean
            self.minimize_energy = minimize_energy
            self.maximize_energy_rec = maximize_energy_rec
            self.hidden_in = hidden_in
            self.visible_in = visible_in
            self.hidden_rec_in = hidden_rec_in
            self.visible_rec_in = visible_rec_in
            self.learning_rate_in = learning_rate_in
            self.summary_enabled = summary_enabled
            self.summaries = summaries
            self.summary_file = summary_file
            self.step = init_step
            self.logger = logging.getLogger()

        def train(self, visible_units, gibbs=1, sigma=1.0, lr=0.0001):
            '''
            one iteration of training one self.crbm
            :param visible_units: numpy array of shape, (b, ) + crbm.visible_shape
            :param gibbs: int. gibb step. has to be greater than 1
            :param sigma: float. noise when reconstructing.
            :param verbose: bool.
            :param lr: float. learning rate
            :return: None.
            '''
            assert gibbs >= 1
            probabilities = self.probabilities
            crbm = self.crbm

            loss = self.loss
            energy_mean = self.energy_mean
            energy_rec_mean = self.energy_rec_mean
            minimize_energy = self.minimize_energy
            maximize_energy_rec = self.maximize_energy_rec
            hidden_in = self.hidden_in
            visible_in = self.visible_in
            hidden_rec_in = self.hidden_rec_in
            visible_rec_in = self.visible_rec_in
            learning_rate_in = self.learning_rate_in
            summary_enabled = self.summary_enabled
            summaries = self.summaries
            summary_file = self.summary_file
            logger = self.logger


            sess = crbm.sess

            hidden = crbm.generate_hidden_units(visible_units)
            # reconstruct
            hidden_rec = hidden
            visible_rec = crbm.generate_visible_units(hidden_rec, sigma=sigma)
            for _ in range(gibbs-1):
                hidden_rec = crbm.generate_hidden_units(visible_rec)
                visible_rec = crbm.generate_visible_units(hidden_rec, sigma=sigma)

            # optimization
            feed_dict = {
                hidden_in: hidden,
                visible_in: visible_units,
                hidden_rec_in: hidden_rec,
                visible_rec_in: visible_rec,
                learning_rate_in: lr
            }
            sess.run(minimize_energy, feed_dict=feed_dict)
            sess.run(maximize_energy_rec, feed_dict=feed_dict)

            # logging.
            log_msg = '''
            error: {0}
            real energy: {1}
            reconstructed energy: {2}
            aggregated hidden units activation probability: {3}
            at step {4}
            '''.format(sess.run(loss, feed_dict=feed_dict), sess.run(energy_mean, feed_dict=feed_dict),
                       sess.run(energy_rec_mean, feed_dict=feed_dict), sess.run(probabilities, feed_dict=feed_dict),
                       self.step)
            logger.debug(log_msg)

            if summary_enabled:
                summary_file.add_summary(sess.run(summaries, feed_dict=feed_dict), global_step=self.step)

            self.step += 1
            return feed_dict

    class Saver:

        @staticmethod
        def save(crbm, path, step):
            '''
            (path='my-model', step=0) ==> filename: 'my-model-0'
            :param crbm: an instance of crbm. to be saved
            :param path: str. path to save.
            :param step: int.
            :return:
            '''
            w = crbm.w
            w_r = crbm.w_r
            vb = crbm.vb
            hb = crbm.hb
            sess = crbm.sess
            with crbm.graph.as_default() as g:
                saver = tf.train.Saver(var_list=[w, vb, hb])
                saver.save(sess, path, global_step=step, write_meta_graph=False)

        @staticmethod
        def restore(crbm, path):
            with crbm.graph.as_default() as g:
                sess = crbm.sess
                tf.train.Saver().restore(sess, path)
            # reset w_r from w. this step might not be needed.
            with crbm.graph.as_default() as g:
                with tf.variable_scope(crbm.params_id) as crbm_scope:
                    crbm.w_r = crbm.w[:,:,::-1]










