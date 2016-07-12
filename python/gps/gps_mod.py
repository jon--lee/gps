""" This file defines the main object that runs experiments. """

import matplotlib as mpl
mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
import csv
import sys
import copy
import argparse
import threading
import time
import numpy as np
import ipdb
import IPython

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps import __file__
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common']) if config['gui_on'] else None

        config['algorithm']['agent'] = self.agent
        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """

        itr_start = self._initialize(itr_load)
        global_cost_counter = []

        for itr in range(itr_start, self._hyperparams['iterations']):
            for cond in self._train_idx:
                for i in range(self._hyperparams['num_samples']):
                    self._take_sample(itr, cond, i)     # <<<<<<<< Look at take_sample

            traj_sample_lists = [
                self.agent.get_samples(cond, -self._hyperparams['num_samples']) # <<<<<<<< Look at get_sample
                for cond in self._train_idx
            ]
            self._take_iteration(itr, traj_sample_lists)  # <<<<<<<< Look at take_iteration
            pol_sample_lists = self._take_policy_samples()
            self._log_data(itr, traj_sample_lists, pol_sample_lists)
            costs = [np.mean(np.sum(self.algorithm.prev[m].cs, axis=1)) for m in range(self.algorithm.M)]
            global_cost_counter += costs       # Stores value in common container 
            print(global_cost_counter)

        return global_cost_counter

    def test_policy(self, itr, N):
        """
        Take N policy samples of the algorithm state at iteration itr,
        for testing the policy to see how it is behaving.
        (Called directly from the command line --policy flag).
        Args:
            itr: the iteration from which to take policy samples
            N: the number of policy samples to take
        Returns: None
        """
        algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr
        self.algorithm = self.data_logger.unpickle(algorithm_file)
        if self.algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1) # called instead of sys.exit(), since t
        traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            ('traj_sample_itr_%02d.pkl' % itr))

        pol_sample_lists = self._take_policy_samples(N)
        self.data_logger.pickle(
            self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
            copy.copy(pol_sample_lists)
        )

        if self.gui:
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.set_status_text(('Took %d policy sample(s) from ' +
                'algorithm state at iteration %d.\n' +
                'Saved to: data_files/pol_sample_itr_%02d.pkl.\n') % (N, itr, itr))

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            if self.gui:
                self.gui.set_status_text('Press \'go\' to begin.')
            return 0
        else:
            algorithm_file = self._data_files_dir + 'algorithm_i_%02d.pkl' % itr_load
            self.algorithm = self.data_logger.unpickle(algorithm_file)
            if self.algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1) # called instead of sys.exit(), since this is in a thread
                
            if self.gui:
                traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('traj_sample_itr_%02d.pkl' % itr_load))
                pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
                    ('pol_sample_itr_%02d.pkl' % itr_load))
                self.gui.update(itr_load, self.algorithm, self.agent,
                    traj_sample_lists, pol_sample_lists)
                self.gui.set_status_text(
                    ('Resuming training from algorithm state at iteration %d.\n' +
                    'Press \'go\' to begin.') % itr_load)
            return itr_load + 1

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        pol = self.algorithm.cur[cond].traj_distr  # Uses a linear gaussian policy
        if self.gui:
            self.gui.set_image_overlays(cond)   # Must call for each new cond.
            redo = True
            while redo:
                while self.gui.mode in ('wait', 'request', 'process'):
                    if self.gui.mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.guigui
                    if self.gui.request == 'reset':
                        try:
                            self.agent.reset(cond)
                        except NotImplementedError:
                            self.gui.err_msg = 'Agent reset unimplemented.'
                    elif self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.gui.set_status_text(
                    'Sampling: iteration %d, condition %d, sample %d.' %
                    (itr, cond, i)
                )
                self.agent.sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()
                    self.agent.delete_last_sample(cond)
                else:
                    redo = False
        else:
            self.agent.sample(
                pol, cond,
                verbose=(i < self._hyperparams['verbose_trials'])
            )

    def _take_iteration(self, itr, sample_lists):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Calculating.')
            self.gui.start_display_calculating()
        self.algorithm.iteration(sample_lists)
        if self.gui:
            self.gui.stop_display_calculating()

    def _take_policy_samples(self, N=None):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            return None
        if not N:
            N = self._hyperparams['verbose_policy_trials']
        if self.gui:
            self.gui.set_status_text('Taking policy samples.')
        pol_samples = [[None for _ in range(N)] for _ in range(self._conditions)]
        for cond in range(len(self._test_idx)):
            for i in range(N):
                pol_samples[cond][i] = self.agent.sample(
                    self.algorithm.policy_opt.policy, self._test_idx[cond],
                    verbose=True, save=False)
        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Logging data and updating GUI.')
            self.gui.update(itr, self.algorithm, self.agent,
                traj_sample_lists, pol_sample_lists)
            self.gui.save_figure(
                self._data_files_dir + ('figure_itr_%02d.png' % itr)
            )
        if 'no_sample_logging' in self._hyperparams['common']:
            return
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        )
        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

    # def _end(self):
    #     """ Finish running and exit. """
    #     if self.gui:
    #         self.gui.set_status_text('Training complete.')
    #         self.gui.end_mode()


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM only)')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.new:
        from shutil import copy

        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)

        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    # if args.targetsetup:
    #     try:
    #         import matplotlib.pyplot as plt
    #         from gps.agent.ros.agent_ros import AgentROS
    #         from gps.gui.target_setup_gui import TargetSetupGUI

    #         agent = AgentROS(hyperparams.config['agent'])
    #         TargetSetupGUI(hyperparams.config['common'], agent)

    #         plt.ioff()
    #         plt.show()
    #     except ImportError:
    #         sys.exit('ROS required for target setup.')
    # elif test_policy_N:
    #     import random
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     random.seed(0)
    #     np.random.seed(0)

    #     data_files_dir = exp_dir + 'data_files/'
    #     data_filenames = os.listdir(data_files_dir)
    #     algorithm_prefix = 'algorithm_itr_'
    #     algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
    #     current_algorithm = sorted(algorithm_filenames, reverse=True)[0]
    #     current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix)+2])

    #     gps = GPSMain(hyperparams.config)
    #     if hyperparams.config['gui_on']:
    #         test_policy = threading.Thread(
    #             target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
    #         )
    #         test_policy.daemon = True
    #         test_policy.start()

    #         plt.ioff()
    #         plt.show()
    #     else:
    #         gps.test_policy(itr=current_itr, N=test_policy_N)
    # else:
    _spawn_and_run_thread(hyperparams_file)     # Launches various threads to comb through different hyperparameters

def _spawn_and_run_thread(hyperparams_file):
    import random
    import matplotlib.pyplot as plt

    random.seed(3)
    np.random.seed(3)

    target_policy_1 = threading.Thread(target=lambda: _agent_set(hyperparams_file), name='hparam1')     # Starts the thread to run batch 1 of hyper params
    target_policy_1.daemon = True

    target_policy_2 = threading.Thread(target=lambda: _algorithm_set(hyperparams_file), name='hparam2')
    target_policy_2.daemon = True
    

    # target_policy_1.start()
    target_policy_2.start()

    # target_policy_1.join()
    target_policy_2.join()

def _agent_set(hyperparams_file):
    try:
        for i in range(1):
            if (i == 0):
                for j in np.arange(0,3,0.1):
                    hyperparams = imp.load_source('hyperparams0', hyperparams_file)
                    hyperparams.agent["rk"] = j
                    gps = GPSMain(hyperparams.config)
                    global_cost_counter = gps.run(itr_load=None)
                    string = "agent # rk # {0}".format(j)
                    _write_to_csv("python/gps/hyperparam_data/agent_set.csv", [string], global_cost_counter)

            if (i == 1):
                for j in np.arange(0,0.2,0.01):
                    hyperparams = imp.load_source('hyperparams0', hyperparams_file)
                    hyperparams.agent["dt"] = j
                    gps = GPSMain(hyperparams.config)
                    global_cost_counter = gps.run(itr_load=None)
                    string = "agent # dt # {0}".format(j)
                    _write_to_csv("python/gps/hyperparam_data/agent_set.csv", [string], global_cost_counter)

            if (i == 2):
                for j in np.arange(1,50,1):
                    hyperparams = imp.load_source('hyperparams0', hyperparams_file)
                    hyperparams.agent["substeps"] = j
                    gps = GPSMain(hyperparams.config)
                    global_cost_counter = gps.run(itr_load=None)
                    string = "agent # substeps # {0}".format(j)
                    _write_to_csv("python/gps/hyperparam_data/agent_set.csv", [string], global_cost_counter)


    except Exception as e:
        print("exception in agent data set")
        # _write_to_csv("python/gps/hyperparam_data/agent_set.csv", [e], [])
        _write_to_csv("python/gps/hyperparam_data/agent_set.csv", ['Error >>>>>'], [])          # >>>>> 5 times to demark the end of the error

def _algorithm_set(hyperparams_file):
    try:
        for i in range(7):
            # if (i == 0):
            #     for j in np.arange(0.1,100,0.5):
            #         hyperparams = imp.load_source('hyperparams1', hyperparams_file)
            #         hyperparams.algorithm["init_traj_distr"]["init_var"] = j
            #         gps = GPSMain(hyperparams.config)
            #         global_cost_counter = gps.run(itr_load=None)
            #         string = "algorithm # init_var # {0}".format(j)                
            #         _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", [string], global_cost_counter)

            # if (i == 1):
            #     for j in np.arange(0.1,100,0.5):
            #         hyperparams = imp.load_source('hyperparams1', hyperparams_file)
            #         hyperparams.algorithm["init_traj_distr"]["stiffness"] = j
            #         gps = GPSMain(hyperparams.config)
            #         global_cost_counter = gps.run(itr_load=None)
            #         string = "algorithm # stiffness # {0}".format(j)                
            #         _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", [string], global_cost_counter)

            # if (i == 2):
            #     weight_list = [(j,k) for j in np.arange(0,2,0.1) for k in np.arange(0,2,0.1)]
            #     for j in weight_list:
            #         hyperparams = imp.load_source('hyperparams1', hyperparams_file)
            #         hyperparams.algorithm["cost"]["weights"] = j
            #         gps = GPSMain(hyperparams.config)
            #         global_cost_counter = gps.run(itr_load=None)
            #         string = "algorithm # weights # {0}".format(j)                
            #         _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", [string], global_cost_counter)

            # if (i == 3):
            #     lst_a       = np.array([10**k for k in range(-7,3)])
            #     lst_b       = lst_a /2 
            #     lst_combine = [c for sublist in zip(lst_b, lst_a) for c in sublist]

            #     for j in lst_combine:
            #         hyperparams = imp.load_source('hyperparams1', hyperparams_file)
            #         hyperparams.algorithm["dynamics"]["regularization"] = j
            #         gps = GPSMain(hyperparams.config)
            #         global_cost_counter = gps.run(itr_load=None)
            #         string = "algorithm # regularization # {0}".format(j)                
            #         _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", [string], global_cost_counter)

            if (i == 4):
                for j in np.arange(2,10,1):
                    hyperparams = imp.load_source('hyperparams1', hyperparams_file)
                    hyperparams.algorithm["dynamics"]["prior"]["max_clusters"] = j
                    gps = GPSMain(hyperparams.config)
                    global_cost_counter = gps.run(itr_load=None)
                    string = "algorithm # max_clusters # {0}".format(j)                
                    _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", [string], global_cost_counter)

            if (i == 5):
                for j in np.arange(1,10,1):
                    hyperparams = imp.load_source('hyperparams1', hyperparams_file)
                    hyperparams.algorithm["dynamics"]["prior"]["min_samples_per_cluster"] = j
                    gps = GPSMain(hyperparams.config)
                    global_cost_counter = gps.run(itr_load=None)
                    string = "algorithm # min_samples_per_cluster # {0}".format(j)                
                    _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", [string], global_cost_counter)

            if (i == 6):
                for j in np.arange(1,10,1):
                    hyperparams = imp.load_source('hyperparams1', hyperparams_file)
                    hyperparams.algorithm["dynamics"]["prior"]["max_samples"] = j
                    gps = GPSMain(hyperparams.config)
                    global_cost_counter = gps.run(itr_load=None)
                    string = "algorithm # max_samples # {0}".format(j)                
                    _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", [string], global_cost_counter)

    except Exception as e:
        print("exception in algorithm data set")
        # _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", [e], [])
        _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", ['Error >>>>>'], [])          # >>>>> 5 times to demark the end of the error

# def _state_cost_set(hyperparams_file):
#     for i in range(1):
#         if (i == 0):
#             weight_list = [(j,k) for j in np.arange(0,2,0.1) for k in np.arange(0,2,0.1)]
#             for j in weight_list:
#                 hyperparams = imp.load_source('hyperparams1', hyperparams_file)
#                 hyperparams.state_cost["data_types"]["init_var"] = j
#                 gps = GPSMain(hyperparams.config)
#                 global_cost_counter = gps.run(itr_load=None)
#                 string = "state_cost # wp # {0}".format(j)                
#                 _write_to_csv("python/gps/hyperparam_data/algorithm_set.csv", [string], global_cost_counter)


def _write_to_csv(file_path, head_msg, global_cost_counter):
    # Function appends to csv with directory given by file_path
    # head_msg is the message that shows what hyperparams are being veried

    # Makes a hyperparams data file to store all the relevant value
    if not os.path.exists('python/gps/hyperparam_data'):
        os.makedirs('python/gps/hyperparam_data')

    try:
        with open(file_path, 'a') as csv_file:
            file_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(head_msg)
            file_writer.writerow(global_cost_counter)
    except:
        print(threading.current_thread().getName() + ' error in write to csv')
        sys.exit(0)

if __name__ == "__main__":
    main()
