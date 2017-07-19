import pathnet
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from game_state import GameState

# from game_ac_network import GameACFFNetwork, GameACLSTMNetwork, GameACPathNetNetwork
from game_ac_network import GameACPathNetNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM
from constants import USE_PATHNET
from constants import ROMZ
from constants import ACTION_SIZEZ

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 training_stage,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device,
                 FLAGS="",
                 task_index=""):

        print("Initializing worker #{}".format(task_index))
        self.training_stage = training_stage
        self.thread_index = thread_index
        self.task_index = task_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        self.local_network = GameACPathNetNetwork(training_stage, thread_index, device,FLAGS)

        self.local_network.prepare_loss(ENTROPY_BETA)

        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            self.gradients = tf.gradients(
                self.local_network.total_loss, var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            self.local_network.get_vars(),
            self.gradients )

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        # variable controling log output
        self.prev_local_t = 0

    def set_training_stage(self, training_stage):
        self.training_stage = training_stage
        self.local_network.set_training_stage(training_stage)
        print("Setting training task to:  " + ROMZ[training_stage] + ", with action size: " + str(ACTION_SIZEZ[self.training_stage]))
        if training_stage == 1:
            self.game_state.close_env()
        self.game_state = GameState(113 * self.task_index, ROMZ[training_stage], display=False, no_op_max=ACTION_SIZEZ[training_stage], task_index=self.task_index)

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score,
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input,score_ph,score_ops, geopath, FLAGS,score_set_ph,score_set_ops):

        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        start_local_t = self.local_t

        res_reward=-1000;
        # t_max times loop
        for i in range(LOCAL_T_MAX):
            pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            action = self.choose_action(pi_)

            states.append(self.game_state.s_t)
            actions.append(action)
            values.append(value_)

            # process game
            self.game_state.process(action)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward

            # clip reward
            rewards.append( np.clip(reward, -1, 1) )

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True
                sess.run(score_ops,{score_ph:self.episode_reward});
                sess.run(score_set_ops,{score_set_ph:self.episode_reward});
                res_reward=self.episode_reward;
                self.episode_reward = 0
                self.game_state.reset()
                if USE_LSTM:
                    self.local_network.reset_state()
                break
        if(res_reward==-1000):
            res_reward=self.episode_reward;
        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game_state.s_t)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + GAMMA * R
            td = R - Vi
            # a = np.zeros([ACTION_SIZEZ[self.training_stage]])
            a = np.zeros([ACTION_SIZEZ[0]])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        var_idx=self.local_network.get_vars_idx();
        gradients_list=[];
        for i in range(len(var_idx)):
            if(var_idx[i]==1.0):
                gradients_list+=[self.apply_gradients[i]];
        sess.run(gradients_list,
                feed_dict = {
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.learning_rate_input: cur_learning_rate} )

        # if (self.training_stage == 0):
        #     sess.run(gradients_list,
        #             feed_dict = {
        #                 self.local_network.s: batch_si,
        #                 self.local_network.a_source: batch_a,
        #                 self.local_network.td: batch_td,
        #                 self.local_network.r: batch_R,
        #                 self.learning_rate_input: cur_learning_rate} )
        # else:
        #     sess.run(gradients_list,
        #             feed_dict = {
        #                 self.local_network.s: batch_si,
        #                 self.local_network.a_target: batch_a,
        #                 self.local_network.td: batch_td,
        #                 self.local_network.r: batch_R,
        #                 self.learning_rate_input: cur_learning_rate} )


        if (self.task_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t,    elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t;
