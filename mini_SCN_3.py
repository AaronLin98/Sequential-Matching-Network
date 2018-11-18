import tensorflow as tf
import pickle
import utils
from keras.preprocessing.sequence import pad_sequences
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, optimized_trilinear_for_attention
import numpy as np
import Evaluate

word_embeddings_path = "../data/embedding.pkl"
char_embeddings_path = "../ubuntu_dict/char_embedd.pkl"
utt_path = "../data/utt.pkl"
utt_char_path = "../ubuntu_dict/utt_char.pkl"
re_path = "../data/re.pkl"
re_char_path = "../ubuntu_dict/re_char.pkl"
ev_path = "../data/ev.pkl"
ev_char_path = "../ubuntu_dict/ev_char.pkl"

def multi_char_sequences_padding(all_sequences, max_sentence_len=50):
    max_num_utterance = 10
    # PAD_SEQUENCE = [0] * max_sentence_len
    PAD_CHAR = [0] * 16
    PAD_CHAR_SEQUENCE = [PAD_CHAR] * 50
    # print(PAD_CHAR_SEQUENCE)
    padded_sequences = []
    for sequences in all_sequences:
        sequences_len = len(sequences)
        if sequences_len < max_num_utterance:
            sequences += [PAD_CHAR_SEQUENCE] * (max_num_utterance - sequences_len)
        else:
            sequences = sequences[-max_num_utterance:]
        sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
        padded_sequences.append(sequences)
    return padded_sequences

class SCN():
    def __init__(self):
        self.max_num_utterance = 10
        self.negative_samples = 1
        self.max_sentence_len = 50
        self.word_embedding_size = 200
        self.char_embedding_dim = 300
        self.wordlen = 16
        self.rnn_units = 200
        self.total_words = 434511 # to do changing
        self.total_chars = 828 # to do changeing 
        self.batch_size = 100

    def LoadModel(self):
        #init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        #with tf.Session() as sess:
            #sess.run(init)
        saver.restore(sess,"model/model.5")
        return sess
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        # with tf.Session() as sess:
        #     # Restore variables from disk.
        #     saver.restore(sess, "/model/model.5")
        #     print("Model restored.")

    def BuildModel(self):
        print("preprocessing build model....")
        # word embedding
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
        self.y_true = tf.placeholder(tf.int32, shape=(None,))
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        self.response_len = tf.placeholder(tf.int32, shape=(None,))
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))
        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words,self.
                                                                      word_embedding_size), dtype=tf.float32, trainable=False)
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)

        # char embedding
        self.response_cph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len, self.wordlen))
        self.embedding_cph = tf.placeholder(tf.float32, shape=(self.total_chars, self.char_embedding_dim))
        char_embeddings = tf.get_variable('char_embeddings_v', shape=(self.total_chars,self.char_embedding_dim), dtype=tf.float32, trainable=False)
        self.char_embeddings_init = char_embeddings.assign(self.embedding_cph)
        response_char_embeddings = tf.nn.embedding_lookup(char_embeddings, self.response_cph)    

        self.utterance_cph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len, self.wordlen))
        all_utterance_ch_embeddings = tf.nn.embedding_lookup(char_embeddings, self.utterance_cph)
        all_utterance_ch_embeddings = tf.unstack(all_utterance_ch_embeddings, num=self.max_num_utterance, axis=1)


        # response : char _ word embedding
        self.N = tf.placeholder(tf.int32,shape=(None))
        d = 96
        dro = 0.1
        self.sample_numbers = tf.placeholder(tf.int32,shape=(None))
        # 2 means (nagetive_samples + 1)
        ch_emb = tf.reshape(response_char_embeddings,[self.sample_numbers * self.N * self.max_sentence_len, self.wordlen, self.char_embedding_dim])
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
        ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
        ch_emb = conv(ch_emb, d, bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = None)
        ch_emb = tf.reduce_max(ch_emb, axis = 1)
        ch_emb = tf.reshape(ch_emb,[self.sample_numbers*self.N, self.max_sentence_len,int(ch_emb.shape[-1])])

        c_emb = tf.nn.dropout(response_embeddings, 1.0 - self.dropout)
        # c_emb = tf.concat([c_emb, ch_emb], axis=2)
        c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)

        # chamge

        A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
        reuse = None

        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, c_emb, sequence_length=self.response_len, dtype=tf.float32,
                                                       scope='sentence_GRU')
        self.response_embedding_save = response_GRU_embeddings
        c_emb = tf.transpose(c_emb, perm=[0, 2, 1])
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        linecounter = 0
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            #  utterance embedding 
            utt_ch_emb = tf.reshape(all_utterance_ch_embeddings[linecounter],[self.sample_numbers*self.N * self.max_sentence_len, self.wordlen, self.char_embedding_dim])
            utt_ch_emb = tf.nn.dropout(utt_ch_emb, 1.0 - 0.5 * self.dropout)
            utt_ch_emb = conv(utt_ch_emb, d, bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = True)
            utt_ch_emb = tf.reduce_max(utt_ch_emb, axis = 1)
            utt_ch_emb = tf.reshape(utt_ch_emb, [self.sample_numbers*self.N, self.max_sentence_len, int(utt_ch_emb.shape[-1])])

            utt_emb = tf.nn.dropout(utterance_embeddings, 1.0 - self.dropout)
            # utt_emb = tf.concat([utt_emb, utt_ch_emb], axis=2)
            utt_emb = highway(utt_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)

            matrix1 = tf.matmul(utt_emb, c_emb)
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, utt_emb, sequence_length=utterance_len, dtype=tf.float32,
                                                            scope='sentence_GRU')
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse, name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
            linecounter += 1

        _, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(matching_vectors, axis=0, name='matching_stack'), dtype=tf.float32,
                                           time_major=True, scope='final_GRU')  # TODO: check time_major
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
        self.y_pred = tf.nn.softmax(logits)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.total_loss)

    def Evaluate(self,sess):
        with open(ev_path, 'rb') as f:
            history, true_utt,labels = pickle.load(f)
        with open(ev_char_path,'rb') as f:
            utt_char,true_ch_utt,ch_labels = pickle.load(f)
        self.all_candidate_scores = []
        history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
        history, history_len = np.array(history), np.array(history_len)
        true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=self.max_sentence_len))
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
        utt_char = multi_char_sequences_padding(utt_char,50)
        true_ch_utt = np.array(pad_sequences(true_ch_utt, padding='post', maxlen=self.max_sentence_len))

        low = 0
        dro = 0.1
        while True:
            feed_dict = {self.utterance_ph: np.concatenate([history[low:low + 200]], axis=0),
                         self.all_utterance_len_ph: np.concatenate([history_len[low:low + 200]], axis=0),
                         self.response_ph: np.concatenate([true_utt[low:low + 200]], axis=0),
                         self.response_len: np.concatenate([true_utt_len[low:low + 200]], axis=0),
                         self.response_cph: np.concatenate([true_ch_utt[low:low + 200]], axis=0),  # todo negs
                         self.utterance_cph: np.concatenate([utt_char[low:low + 200]], axis=0),
                         self.dropout: dro,
                         self.N: 200,
                         self.sample_numbers: 1
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + 200
            if low >= history.shape[0]:
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)
        computeR10_1 = Evaluate.ComputeR10_1(all_candidate_scores,labels)
        computeR2_1 = Evaluate.ComputeR2_1(all_candidate_scores,labels)
        return computeR10_1,computeR2_1


    def TrainModel(self,countinue_train = False, previous_modelpath = "model"):
        print("preprocessing .. train model ")
        f_write_loss = open('result_log_2.txt','w')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("output_mini", sess.graph)
            train_writer = tf.summary.FileWriter('output_mini', sess.graph)
            print("open file .....")
            with open(re_path, 'rb') as f:
                actions = pickle.load(f)
            with open(word_embeddings_path, 'rb') as f:
                embeddings = pickle.load(f,encoding="bytes")
            with open(char_embeddings_path, 'rb') as f:
                charembeddings = pickle.load(f,encoding="bytes")
            with open(utt_path, 'rb') as f:
               history, true_utt = pickle.load(f)
            with open(re_char_path, 'rb') as f:
                ch_actions = pickle.load(f)
            with open(utt_char_path,'rb') as f:
                utt_char, true_ch_utt = pickle.load(f)
            # with open("data/biglearn_test_small.txt", encoding="utf8") as f:
            #     lines = f.readlines()
            #     history, true_utt = utils.build_evaluate_data(lines)
            print("np file padding .....")
            history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
            utt_char = multi_char_sequences_padding(utt_char,50)
            true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=self.max_sentence_len))
            true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
            true_ch_utt = np.array(pad_sequences(true_ch_utt, padding='post', maxlen=self.max_sentence_len))
            actions_len = np.array(utils.get_sequences_length(actions, maxlen=self.max_sentence_len))
            actions = np.array(pad_sequences(actions, padding='post', maxlen=self.max_sentence_len))
            ch_actions = np.array(pad_sequences(ch_actions, padding='post', maxlen=self.max_sentence_len))
            history, history_len = np.array(history), np.array(history_len)
            if countinue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: embeddings})
                sess.run(self.char_embeddings_init, feed_dict={self.embedding_cph: charembeddings})
            else:
                saver.restore(sess,previous_modelpath)
            low = 0
            epoch = 1
            dro = 0.1
            print("starting epoch ....")
            while epoch < 10:
                n_sample = min(low + self.batch_size, history.shape[0]) - low
                # sess.run(self.N, feed_dict={self.N: n_sample})
                negative_indices = [np.random.randint(0, actions.shape[0], n_sample) for _ in range(self.negative_samples)]
                negs = [actions[negative_indices[i], :] for i in range(self.negative_samples)]
                negs_ch = [ch_actions[negative_indices[i], :] for i in range(self.negative_samples)]
                negs_len = [actions_len[negative_indices[i]] for i in range(self.negative_samples)]
                # print("feed_dict ......")
                feed_dict = {self.utterance_ph: np.concatenate([history[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                             self.all_utterance_len_ph: np.concatenate([history_len[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                             self.response_ph: np.concatenate([true_utt[low:low + n_sample]] + negs, axis=0),
                             self.response_len: np.concatenate([true_utt_len[low:low + n_sample]] + negs_len, axis=0),
                             self.y_true: np.concatenate([np.ones(n_sample)] + [np.zeros(n_sample)] * self.negative_samples, axis=0),
                             self.response_cph: np.concatenate([true_ch_utt[low:low + n_sample]] + negs_ch, axis=0),  # todo negs
                             self.utterance_cph: np.concatenate([utt_char[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                             self.dropout: dro,
                             self.N: n_sample,
                             self.sample_numbers: self.negative_samples + 1
                             }
                # print("starting run model .....")
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                if low % 102400 == 0:
                    # print("loss",sess.run(self.total_loss, feed_dict=feed_dict))
                    w_loss = sess.run(self.total_loss, feed_dict=feed_dict)
                    print(w_loss)
                    f_write_loss.write("loss: ")
                    f_write_loss.write(str(w_loss))
                    f_write_loss.write("\n")
                    computeR10_1,computeR2_1 = self.Evaluate(sess)
                    # print(computeR10_1)
                    # print(computeR2_1)
                    f_write_loss.write("computeR10_1: ")
                    f_write_loss.write(str(computeR10_1))
                    f_write_loss.write("\n")
                    f_write_loss.write("computeR2_1: ")
                    f_write_loss.write(str(computeR2_1))
                    f_write_loss.write("\n")
                    # print("eva end......")
                    # break
                if low >= history.shape[0]:
                    low = 0
                    saver.save(sess,"model_1/model.{0}".format(epoch))
                    # print(sess.run(self.total_loss, feed_dict=feed_dict))
                    h_loss = sess.run(self.total_loss, feed_dict=feed_dict)
                    print(h_loss)
                    f_write_loss.write("h_loss: ")
                    f_write_loss.write(str(h_loss))
                    f_write_loss.write("\n")
                    print('epoch={i}'.format(i=epoch))
                    f_write_loss.write("epoch: ")
                    f_write_loss.write(str(epoch))
                    f_write_loss.write("\n")
                    epoch += 1
        f_write_loss.close()

if __name__ == "__main__":
    print("initial......")
    scn =SCN()
    scn.BuildModel()
    scn.TrainModel()
    # sess = scn.LoadModel()
    # scn.Evaluate(sess)
    #results = scn.BuildIndex(sess)
    #print(len(results))

    #scn.TrainModel()
