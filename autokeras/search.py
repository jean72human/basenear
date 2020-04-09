import logging
import os
import queue
import re
import sys
import time
from datetime import datetime
from abc import abstractmethod
from random import randrange
from multiprocessing import Process
from multiprocessing import Pool, Queue

import torch
import torch.multiprocessing as mp

from autokeras.bayesian import BayesianOptimizer, SearchTree, contain
from autokeras.greedy import GreedyOptimizer
from autokeras.constant import Constant
from autokeras.nn.model_trainer import ModelTrainer
from autokeras.net_transformer import transform
from autokeras.utils import pickle_to_file, pickle_from_file, verbose_print, get_system, assert_search_space




class Searcher:
    """The base class to search for neural architectures.

    This class generate new architectures, call the trainer to train it, and update the optimizer.

    Attributes:
        n_classes: Number of classes in the target classification task.
        input_shape: Arbitrary, although all dimensions in the input shaped must be fixed.
            Use the keyword argument `input_shape` (tuple of integers, does not include the batch axis)
            when using this layer as the first layer in a model.
        verbose: Verbosity mode.
        history: A list that stores the performance of model. Each element in it is a dictionary of 'model_id',
            'loss', and 'metric_value'.
        neighbour_history: A list that stores the performance of neighbor of the best model.
            Each element in it is a dictionary of 'model_id', 'loss', and 'metric_value'.
        path: A string. The path to the directory for saving the searcher.
        metric: An instance of the Metric subclasses.
        loss: A function taking two parameters, the predictions and the ground truth.
        generators: A list of generators used to initialize the search.
        model_count: An integer. the total number of neural networks in the current searcher.
        descriptors: A dictionary of all the neural network architectures searched.
        trainer_args: A dictionary. The params for the constructor of ModelTrainer.
        default_model_len: An integer. Number of convolutional layers in the initial architecture.
        default_model_width: An integer. The number of filters in each layer in the initial architecture.
        training_queue: A list of the generated architectures to be trained.
        x_queue: A list of trained architectures not updated to the gpr.
        y_queue: A list of trained architecture performances not updated to the gpr.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose,               
                 trainer_args=None,
                 default_model_len=None,
                 default_model_width=None,
                 n_parralel = 1):
        """Initialize the Searcher.

        Args:
            n_output_node: An integer, the number of classes.
            input_shape: A tuple. e.g. (28, 28, 1).
            path: A string. The path to the directory to save the searcher.
            metric: An instance of the Metric subclasses.
            loss: A function taking two parameters, the predictions and the ground truth.
            generators: A list of generators used to initialize the search.
            verbose: A boolean. Whether to output the intermediate information to stdout.
            trainer_args: A dictionary. The params for the constructor of ModelTrainer.
            default_model_len: An integer. Number of convolutional layers in the initial architecture.
            default_model_width: An integer. The number of filters in each layer in the initial architecture.
        """
        if trainer_args is None:
            trainer_args = {}
        self.n_classes = n_output_node
        self.input_shape = input_shape
        self.verbose = verbose
        self.history = []
        self.neighbour_history = []
        self.path = path
        self.metric = metric
        self.loss = loss
        self.generators = generators
        self.model_count = 0
        self.descriptors = []
        self.trainer_args = trainer_args
        self.default_model_len = default_model_len if default_model_len is not None else Constant.MODEL_LEN
        self.default_model_width = default_model_width if default_model_width is not None else Constant.MODEL_WIDTH
        if 'max_iter_num' not in self.trainer_args:
            self.trainer_args['max_iter_num'] = Constant.SEARCH_MAX_ITER

        self.training_queue = []
        self.x_queue = []
        self.y_queue = []
        logging.basicConfig(filename=os.path.join(self.path, datetime.now().strftime('run_%d_%m_%Y : _%H_%M.log')),
                            format='%(asctime)s - %(filename)s - %(message)s', level=logging.DEBUG)

        self._timeout = None
        
        self.n_parralel = n_parralel

    def load_model_by_id(self, model_id):
        return pickle_from_file(os.path.join(self.path, str(model_id) + '.graph'))

    def load_best_model(self):
        return self.load_model_by_id(self.get_best_model_id())

    def get_metric_value_by_id(self, model_id):
        for item in self.history:
            if item['model_id'] == model_id:
                return item['metric_value']
        return None

    def get_best_model_id(self):
        if self.metric.higher_better():
            return max(self.history, key=lambda x: x['metric_value'])['model_id']
        return min(self.history, key=lambda x: x['metric_value'])['model_id']

    def replace_model(self, graph, model_id):
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.graph'))

    def init_search(self):
        """Call the generators to generate the initial architectures for the search."""
        if self.verbose:
            print('\nInitializing search.')
        for generator in self.generators:
            graph = generator(self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())

        if self.verbose:
            print('Initialization finished.')

    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        """Run the search loop of training, generating and updating once.

        The function will run the training and generate in parallel.
        Then it will update the controller.
        The training is just pop out a graph from the training_queue and train it.
        The generate will call the self.generate function.
        The update will call the self.update function.

        Args:
            train_data: An instance of DataLoader.
            test_data: An instance of Dataloader.
            timeout: An integer, time limit in seconds.
        """
        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()
        mpq = Queue(self.n_parralel * 85)
        self._timeout = time.time() + timeout if timeout is not None else sys.maxsize
        self.trainer_args['timeout'] = timeout
        # Start the new process for training.
        
        if len(self.training_queue) < 1:
            search_results = self._search_common()
            if self.verbose and search_results:
                for (generated_graph, generated_other_info, new_model_id) in search_results:
                    verbose_print(generated_other_info, generated_graph, new_model_id)
        
        if (self.n_parralel > 1):
            print("TRAINING ",len(self.training_queue)," MODELS IN PARRALEL")
        processes = []

        for i in range(min(self.n_parralel,len(self.training_queue))):
            graph, other_info, model_id = self.training_queue.pop(0)
            if self.verbose:
                print('\n')
                print('+' + '-' * 46 + '+')
                print('|' + 'Training model {}'.format(model_id).center(46) + '|')
                print('+' + '-' * 46 + '+')

            #self.sp_search(graph, other_info, model_id, train_data, test_data)
            p = Process(target=self.sp_search, args=(graph, other_info, model_id, train_data, test_data, mpq))
            p.start()
            processes.append(p)

        #for proc in processes:
        #    proc.start()             
        for proc in processes:
            proc.join()
        while not mpq.empty():
            metric_value,loss,model_id,other_info = mpq.get()
            self.add_model(metric_value, loss, model_id)
            graph = pickle_from_file(os.path.join(self.path, str(model_id) + '.graph'))
            self.update(other_info, model_id, graph, metric_value)                 
            
        

    def sp_search(self, graph, other_info, model_id, train_data, test_data, mpq):
        try:
            metric_value, loss, graph = train(None, graph, train_data, test_data, self.trainer_args,
                                              self.metric, self.loss, self.verbose, self.path)

            if metric_value is not None:
                self.write_graph(graph,model_id)
                mpq.put((metric_value,loss,model_id,other_info))
                #self.add_model(metric_value, loss, graph, model_id)
                #self.update(other_info, model_id, graph, metric_value)

        except TimeoutError as e:
            raise TimeoutError from e

    def _search_common(self, mp_queue=None):
        search_results = []
        if not self.training_queue:
            results = self.generate(mp_queue)
            for (generated_graph, generated_other_info) in results:
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((generated_graph, generated_other_info, new_model_id))
                self.descriptors.append(generated_graph.extract_descriptor())
                search_results.append((generated_graph, generated_other_info, new_model_id))
            self.neighbour_history = []

        return search_results

    @abstractmethod
    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            list of 2-element tuples: generated_graph and other_info,
            generated_graph: An instance of Graph.
            other_info: Anything to be saved in the training queue together
            with the architecture.
        """
        pass

    @abstractmethod
    def update(self, *args):
        pass

    def write_graph(self, graph, model_id):
        graph.clear_operation_history()
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.graph'))


    def add_model(self, metric_value, loss, model_id):
        """Append the information of evaluated architecture to history."""
        if self.verbose:
            print('\nSaving model.')

        ret = {'model_id': model_id, 'loss': loss, 'metric_value': metric_value}
        self.neighbour_history.append(ret)
        self.history.append(ret)

        # Update best_model text file
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, 'best_model.txt'), 'w')
            file.write('best model: ' + str(model_id))
            file.close()

        if self.verbose:
            idx = ['model_id', 'loss', 'metric_value']
            header = ['Model ID', 'Loss', 'Metric Value']
            line = '|'.join(x.center(24) for x in header)
            print('+' + '-' * len(line) + '+')
            print('|' + line + '|')

            if self.history:
                r = self.history[-1]
                print('+' + '-' * len(line) + '+')
                line = '|'.join(str(r[x]).center(24) for x in idx)
                print('|' + line + '|')
            print('+' + '-' * len(line) + '+')

        return ret


class BayesianSearcher(Searcher):
    """ Class to search for neural architectures using Bayesian search strategy.

    Attribute:
        optimizer: An instance of BayesianOptimizer.
        t_min: A float. The minimum temperature during simulated annealing.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss,
                 generators, verbose, trainer_args=None,
                 default_model_len=None, default_model_width=None,
                 t_min=None):
        super(BayesianSearcher, self).__init__(n_output_node, input_shape,
                                               path, metric, loss,
                                               generators, verbose,
                                               trainer_args,
                                               default_model_len,
                                               default_model_width)
        if t_min is None:
            t_min = Constant.T_MIN
        self.optimizer = BayesianOptimizer(self, t_min, metric)

    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            list of 2-element tuples: generated_graph and other_info,
            for bayesian searcher the length of list is 1.
            generated_graph: An instance of Graph.
            other_info: Anything to be saved in the training queue together with the architecture.

        """
        remaining_time = self._timeout - time.time()
        generated_graph, new_father_id = self.optimizer.generate(self.descriptors,
                                                                 remaining_time, multiprocessing_queue)
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)

        return [(generated_graph, new_father_id)]

    def update(self, other_info, model_id, graph, metric_value):
        """ Update the controller with evaluation result of a neural architecture.

        Args:
            other_info: Anything. In our case it is the father ID in the search tree.
            model_id: An integer.
            graph: An instance of Graph. The trained neural architecture.
            metric_value: The final evaluated metric value.
        """
        father_id = other_info
        self.optimizer.fit([graph.extract_descriptor()], [metric_value])
        self.optimizer.add_child(father_id, model_id)
        
class ParralelSearcher(Searcher):
    """ Class to search for neural architectures using Bayesian search strategy.

    Attribute:
        optimizer: An instance of BayesianOptimizer.
        t_min: A float. The minimum temperature during simulated annealing.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss,
                 generators, verbose, trainer_args=None,
                 default_model_len=None, default_model_width=None,
                 t_min=None, n_parralel=1):
        super(ParralelSearcher, self).__init__(n_output_node, input_shape,
                                               path, metric, loss,
                                               generators, verbose,
                                               trainer_args,
                                               default_model_len,
                                               default_model_width,
                                               n_parralel)
        if t_min is None:
            t_min = Constant.T_MIN
        self.optimizer = BayesianOptimizer(self, t_min, metric)
        self.std_op = BayesianOptimizer(self, t_min, metric)
        
        
    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            list of 2-element tuples: generated_graph and other_info,
            for bayesian searcher the length of list is 1.
            generated_graph: An instance of Graph.
            other_info: Anything to be saved in the training queue together with the architecture.

        """
        returns = []
        to_be_added = max(self.n_parralel - len(self.training_queue),0)
        first_was_generated = False
        
        while len(returns) < to_be_added:
            if not first_was_generated:
                remaining_time = self._timeout - time.time()
                generated_graph, new_father_id = self.optimizer.generate(self.descriptors,
                                                                         remaining_time, multiprocessing_queue)

            else:
                generated_graph, new_father_id = self.std_op.generate_from_std(self.descriptors,
                                                                     remaining_time, multiprocessing_queue)  
            
            if new_father_id is None:
                    new_father_id = 0
                    generated_graph = self.generators[0](self.n_classes, self.input_shape).generate(self.default_model_len, self.default_model_width)
                    
            self.std_op.fit([generated_graph.extract_descriptor()], [0])
            returns.append((generated_graph, new_father_id))
          
        return returns

    def update(self, other_info, model_id, graph, metric_value):
        """ Update the controller with evaluation result of a neural architecture.

        Args:
            other_info: Anything. In our case it is the father ID in the search tree.
            model_id: An integer.
            graph: An instance of Graph. The trained neural architecture.
            metric_value: The final evaluated metric value.
        """
        father_id = other_info
        self.optimizer.fit([graph.extract_descriptor()], [metric_value])
        self.optimizer.add_child(father_id, model_id)


class GreedySearcher(Searcher):
    """ Class to search for neural architectures using Greedy search strategy.

    Attribute:
        optimizer: An instance of BayesianOptimizer.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose,
                 trainer_args=None,
                 default_model_len=None,
                 default_model_width=None):
        super(GreedySearcher, self).__init__(n_output_node, input_shape,
                                             path, metric, loss, generators,
                                             verbose, trainer_args, default_model_len,
                                             default_model_width)
        self.optimizer = GreedyOptimizer(self, metric)

    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.
                pass into the search algorithm for synchronizing

        Returns:
            results: A list of 2-element tuples. Each tuple contains an instance of Graph,
                and anything to be saved in the training queue together with the architecture

        """
        remaining_time = self._timeout - time.time()
        results = self.optimizer.generate(self.descriptors, remaining_time,
                                          multiprocessing_queue)
        if not results:
            new_father_id = 0
            generated_graph = self.generators[0](self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)
            results.append((generated_graph, new_father_id))

        return results

    def update(self, other_info, model_id, *args):
        """ Update the controller with evaluation result of a neural architecture.

        Args:
            other_info: Anything. In our case it is the father ID in the search tree.
            model_id: An integer.
        """
        father_id = other_info
        self.optimizer.add_child(father_id, model_id)

    def load_neighbour_best_model(self):
        return self.load_model_by_id(self.get_neighbour_best_model_id())

    def get_neighbour_best_model_id(self):
        if self.metric.higher_better():
            return max(self.neighbour_history, key=lambda x: x['metric_value'])['model_id']
        return min(self.neighbour_history, key=lambda x: x['metric_value'])['model_id']


class GridSearcher(Searcher):
    """ Class to search for neural architectures using Greedy search strategy.

    Attribute:
        search_space: A dictionary. Specifies the search dimensions and their possible values
    """
    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose, search_space={},
                 trainer_args=None, default_model_len=None, default_model_width=None):
        super(GridSearcher, self).__init__(n_output_node, input_shape, path, metric, loss, generators, verbose,
                                           trainer_args, default_model_len, default_model_width)
        self.search_space, self.search_dimensions = assert_search_space(search_space)
        self.search_space_counter = 0

    def get_search_dimensions(self):
        return self.search_dimensions

    def search_space_exhausted(self):
        """ Check if Grid search has exhausted the search space """
        if self.search_space_counter == len(self.search_dimensions):
            return True
        return False

    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        """Run the search loop of training, generating and updating once.

        Call the base class implementation for search with

        Args:
            train_data: An instance of DataLoader.
            test_data: An instance of Dataloader.
            timeout: An integer, time limit in seconds.
        """
        if self.search_space_exhausted():
            return
        else:
            super().search(train_data, test_data, timeout)

    def update(self, other_info, model_id, graph, metric_value):
        return

    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            list of 2-element tuples: generated_graph and other_info,
            for grid searcher the length of list is 1.
            generated_graph: An instance of Graph.
            other_info: Always 0.
        """
        grid = self.get_grid()
        self.search_space_counter += 1
        generated_graph = self.generators[0](self.n_classes, self.input_shape). \
            generate(grid[Constant.LENGTH_DIM], grid[Constant.WIDTH_DIM])
        return [(generated_graph, 0)]

    def get_grid(self):
        """ Return the next grid to be searched """
        if self.search_space_counter < len(self.search_dimensions):
            return self.search_dimensions[self.search_space_counter]
        return None


class RandomSearcher(Searcher):
    """ Class to search for neural architectures using Random search strategy.
    Attributes:
        search_tree: The network morphism search tree
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose,
                 trainer_args=None,
                 default_model_len=None,
                 default_model_width=None):
        super(RandomSearcher, self).__init__(n_output_node, input_shape,
                                             path, metric, loss, generators,
                                             verbose, trainer_args, default_model_len,
                                             default_model_width)
        self.search_tree = SearchTree()

    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.

        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.

        Returns:
            list of 2-element tuples: generated_graph and other_info,
            for random searcher the length of list is 1.
            generated_graph: An instance of Graph.
            other_info: Anything to be saved in the training queue together with the architecture.

        """
        random_index = randrange(len(self.history))
        model_id = self.history[random_index]['model_id']
        graph = self.load_model_by_id(model_id)
        new_father_id = None
        generated_graph = None
        for temp_graph in transform(graph):
            if not contain(self.descriptors, temp_graph.extract_descriptor()):
                new_father_id = model_id
                generated_graph = temp_graph
                break
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)

        return [(generated_graph, new_father_id)]

    def update(self, other_info, model_id, *args):
        """ Update the controller with evaluation result of a neural architecture.

        Args:
            other_info: Anything. In our case it is the father ID in the search tree.
            model_id: An integer.
        """
        father_id = other_info
        self.search_tree.add_child(father_id, model_id)


def train(q, graph, train_data, test_data, trainer_args, metric, loss, verbose, path):
    """Train the neural architecture."""
    try:
        model = graph.produce_model()
        loss, metric_value = ModelTrainer(model=model,
                                          path=path,
                                          train_data=train_data,
                                          test_data=test_data,
                                          metric=metric,
                                          loss_function=loss,
                                          verbose=verbose).train_model(**trainer_args)
        model.set_weight_to_graph()
        if q:
            q.put((metric_value, loss, model.graph))
        return metric_value, loss, model.graph
    except RuntimeError as e:
        if not re.search('out of memory', str(e)):
            raise e
        if verbose:
            print('\nCurrent model size is too big. Discontinuing training this model to search for other models.')
        Constant.MAX_MODEL_SIZE = graph.size() - 1
        if q:
            q.put((None, None, None))
        return None, None, None
    except TimeoutError:
        if q:
            q.put((None, None, None))
        return None, None, None