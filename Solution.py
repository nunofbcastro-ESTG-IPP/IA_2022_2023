import copy
import random
import sys
from enum import Enum
from Config import *
from Utils import *

class SolutionType(Enum):
    DEEP_LEARNING = 'Deep Learning'
    RANDOM_FOREST = 'Random Forest'
    GRADIENT_BOOSTING_MACHINE = 'Gradient Boosting Machine'

class Item:
    def __init__(self, seed):
        self.seed = seed  
    
    @staticmethod
    def generate_seed():
        return random.choice(
            rangeInt(
                1, 
                int(sys.maxsize / (10**random.randint(0, len(str(sys.maxsize))-1))), 
                Config.get_step_seed()
            )
        )     
    
    def __eq__(self, item):
        if not isinstance(item, type(self)):
            return False

        self_attributes = vars(self)
        item_attributes = vars(item)

        return self_attributes == item_attributes
    
    def __hash__(self):
        attributes = vars(self)
        attribute_values = []
        for value in attributes.values():
            if isinstance(value, list):
                attribute_values.append(tuple(value))
            else:
                attribute_values.append(value)
        return hash(tuple(attribute_values))

    def __str__(self) -> str:
        attributes = vars(self)
        attribute_strs = [f"{key} = {str(value)}" for key, value in attributes.items()]
        return ", ".join(attribute_strs)

class ItemRandomForest(Item):
    def __init__(self, ntrees, max_depth, min_rows, sample_rate, seed, nfolds = 0, stopping_tolerance = Config.get_gbm_stopping_tolerance(), stopping_rounds = Config.get_gbm_stopping_rounds()) -> None:
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.sample_rate = sample_rate
        super().__init__(seed)
        self.nfolds = nfolds
        self.stopping_tolerance = stopping_tolerance
        self.stopping_rounds = stopping_rounds        
    
    @staticmethod
    def generate_ntrees():
        return random.choice(
            Config.get_rf_range_ntrees()
        )    
    
    @staticmethod
    def generate_max_depth():
        return random.choice(
            Config.get_rf_range_max_depth()
        )

    @staticmethod
    def generate_min_rows():
        return random.choice(
            Config.get_rf_step_min_rows()
        )
    
    @staticmethod
    def generate_sample_rate():
        return random.choice(
            Config.get_rf_sample_rate()
        ) / 100
    
    @staticmethod
    def generate_item():
        return ItemRandomForest(
            ntrees = ItemRandomForest.generate_ntrees(),
            max_depth = ItemRandomForest.generate_max_depth(),
            min_rows = ItemRandomForest.generate_min_rows(),
            sample_rate = ItemRandomForest.generate_sample_rate(),
            seed = ItemRandomForest.generate_seed(),
        )

class ItemDeepLearning(Item):
    def __init__(self, hidden, epochs, mini_batch_size, rate, activation, seed, adaptive_rate=False, stopping_tolerance = Config.get_dl_stopping_tolerance(), stopping_rounds = Config.get_dl_stopping_rounds()) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.rate = rate
        self.activation = activation
        super().__init__(seed)
        self.adaptive_rate = adaptive_rate
        self.stopping_tolerance = stopping_tolerance
        self.stopping_rounds = stopping_rounds
    
    @staticmethod
    def generate_hidden():        
        hiddenRange = Config.get_dl_range_hidden()
        
        hidden = [random.choice(hiddenRange) for _ in range(random.choice(
            Config.get_dl_range_size_hidden()
        ))]
        
        return hidden
    
    @staticmethod
    def generate_epochs():
        return random.choice(
            Config.get_dl_range_epochs()
        )
    
    @staticmethod
    def generate_mini_batch_size():
        return random.choice(
            Config.get_dl_range_mini_batch_size()
        )
    
    @staticmethod
    def generate_rate():
        return random.choice(
            Config.get_dl_range_rate()
        )
    
    @staticmethod
    def generate_activation():
        return random.choice(
            ['tanh', 'tanh_with_dropout', 'rectifier', 'rectifier_with_dropout']
        )
    
    @staticmethod
    def generate_item():
        return ItemDeepLearning(
            hidden = ItemDeepLearning.generate_hidden(),
            epochs = ItemDeepLearning.generate_epochs(),
            mini_batch_size = ItemDeepLearning.generate_mini_batch_size(),
            rate = ItemDeepLearning.generate_rate(),
            activation = ItemDeepLearning.generate_activation(),
            seed = ItemDeepLearning.generate_seed(),
        )

class ItemGradientBoostingMachine(Item):
    def __init__(self, ntrees, max_depth, min_rows, sample_rate, col_sample_rate, learn_rate, seed, nfolds = 0, stopping_tolerance = Config.get_gbm_stopping_tolerance(), stopping_rounds = Config.get_gbm_stopping_rounds()) -> None:
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.sample_rate = sample_rate
        self.col_sample_rate = col_sample_rate
        self.learn_rate = learn_rate
        super().__init__(seed)
        self.nfolds = nfolds
        self.stopping_tolerance = stopping_tolerance
        self.stopping_rounds = stopping_rounds
    
    @staticmethod
    def generate_ntrees():
        return random.choice(
            Config.get_gbm_range_ntrees()
        )
    
    @staticmethod
    def generate_max_depth():
        return random.choice(
            Config.get_gbm_range_max_depth()
        )
    
    @staticmethod
    def generate_min_rows():
        return random.choice(
            Config.get_gbm_range_min_rows()
        )
        
    @staticmethod
    def generate_sample_rate():
        return random.choice(
            Config.get_gbm_range_sample_rate()
        ) / 100
        
    @staticmethod
    def generate_col_sample_rate():
        return random.choice(
            Config.get_gbm_range_col_sample_rate()
        ) / 100
    
    @staticmethod
    def generate_learn_rate():
        return random.choice(
            Config.get_gbm_range_learn_rate()
        )
    
    @staticmethod
    def generate_item():
        return ItemGradientBoostingMachine(
            ntrees = ItemGradientBoostingMachine.generate_ntrees(),
            max_depth = ItemGradientBoostingMachine.generate_max_depth(),
            min_rows = ItemGradientBoostingMachine.generate_min_rows(),
            sample_rate = ItemGradientBoostingMachine.generate_sample_rate(),
            col_sample_rate = ItemGradientBoostingMachine.generate_col_sample_rate(),
            learn_rate = ItemGradientBoostingMachine.generate_learn_rate(),
            seed = ItemGradientBoostingMachine.generate_seed()
        )


class Solution:
    def __init__(self, item) -> None:
        if self._getType(item) == None:
            raise TypeError("O argumento 'item' deve ser uma instância de 'Item'.")
        
        self.item = copy.deepcopy(item)  

        self.fitness = None
    
    @staticmethod
    def _getType(item) -> SolutionType:
        switcher = {
            ItemDeepLearning: SolutionType.DEEP_LEARNING,
            ItemRandomForest: SolutionType.RANDOM_FOREST,
            ItemGradientBoostingMachine: SolutionType.GRADIENT_BOOSTING_MACHINE,
        }
        return switcher.get(type(item), None)
    
    def getType(self) -> SolutionType:
        return self._getType(self.item)

    def __hash__(self):
        return hash((self.getType(), self.item))
    
    def __eq__(self, solution):
        if not isinstance(solution, Solution):
            return False

        return self.item == solution.item
    
    def __str__(self) -> str:
        return f'Type: {self.getType().value} | Configuration: {self.item.__str__()} | Fitness: {str(self.fitness)}'
