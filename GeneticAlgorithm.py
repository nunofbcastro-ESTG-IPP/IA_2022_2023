from Charts import Charts
from Config import *
from Files import *
from MachineLearning import MachineLearning
from Metrics import Metrics
from Solution import *
from Utils import rangeInt
import math


class GeneticAlgorithm:    
    def __init__(self):
        #DownloadDataset()
        self.MachineLearning = MachineLearning(
            file_path = Path(file = 'train.csv', folder = 'datasets'), 
            data_percentage_use=Config.get_data_percentage_use_in_ga(),
            columnNamesFirst = 2, 
            columnNamesLast = None, 
            columnResult = 'label', 
            train_ratio = Config.get_train_ratio(), 
            test_ratio = Config.get_test_ratio(), 
            seed = None
        )
    
    def init_pop(self):
        popArray = divide_number(Config.get_pop_size(), 3)
        
        popModels= {}
        popModels[SolutionType.DEEP_LEARNING]=[Solution(ItemDeepLearning.generate_item()) for _ in range(popArray[0])]
        popModels[SolutionType.RANDOM_FOREST]=[Solution(ItemRandomForest.generate_item()) for _ in range(popArray[1])]  
        popModels[SolutionType.GRADIENT_BOOSTING_MACHINE]=[Solution(ItemGradientBoostingMachine.generate_item()) for _ in range(popArray[2])]
        
        return popModels

    def print_pop(self, pop):
        for sol in pop:
            print(sol.__str__())
    
    def calculate_fitness(self, metrics: Metrics) -> float:
        # Apply the weights to each performance metric
        weighted_mse = Config.get_fitness_mse_percent() * metrics.mse
        weighted_rmse = Config.get_fitness_rmse_percent() * metrics.rmse
        weighted_logloss = Config.get_fitness_logloss_percent() * metrics.logloss
        weighted_mean_per_class_error = Config.get_fitness_mean_per_class_error_percent() * metrics.mean_per_class_error
        # We want to minimize (1 - r2) to maximize r2
        weighted_r2 = Config.get_fitness_r2_percent() * (1 - metrics.r2)
        # We want to minimize (1 - accuracy) to maximize accuracy
        weighted_accuracy = Config.get_fitness_accuracy_percent() * (1 - metrics.accuracy)

        # Calculate the fitness
        fitness = 1 / (1 + weighted_mse + weighted_rmse + weighted_logloss + weighted_mean_per_class_error + weighted_r2 + weighted_accuracy)

        return fitness
    
    def get_model(self, sol: Solution):
        switcher = {
            SolutionType.DEEP_LEARNING: self.MachineLearning.get_model_deep_learning,
            SolutionType.RANDOM_FOREST: self.MachineLearning.get_model_random_forest,
            SolutionType.GRADIENT_BOOSTING_MACHINE: self.MachineLearning.get_model_gradient_boosting_estimator,
        }

        return switcher.get(sol.getType(), None)(solution=sol)
    
    def fitness(self, sol: Solution):
        try:
            # Train model
            model = self.get_model(sol)

            metrics = self.MachineLearning.get_model_performance(model=model)

            sol.fitness = self.calculate_fitness(metrics) 
        except:
            sol.fitness = 0
    
    def fitness_pop(self, pop, genNumber):
        for c, sol in enumerate(pop):
            print(f"{genNumber} - {c} - {sol.getType().value}")
            if sol.fitness == None:
                self.fitness(sol)
            print(sol.fitness)
    
    def get_variables(self, sol: Solution):
        variables = []
        for variable in vars(sol.item):
            generate_function = getattr(sol.item, f"generate_{variable}", None)
            if generate_function is not None and callable(generate_function):
                variables.append(variable)
        return variables

    def mutate(self, sol: Solution):
        new_sol = copy.deepcopy(sol)
        new_sol.fitness = None
        
        variables = self.get_variables(sol) 
        num_parameters = len(variables)
        num_parameters_changing = random.choice(rangeInt(1, int(num_parameters * Config.get_max_prec_mutation_replace_params_values())))
        variables_changing = random.sample(variables, num_parameters_changing)
        
        for variable in variables_changing:   
            setattr(new_sol.item, variable, getattr(new_sol.item, f"generate_{variable}")())
            
        return new_sol
    
    def crossover(self, sol1: Solution, sol2: Solution):    
        new_sol1 = copy.deepcopy(sol1)
        new_sol1.fitness = None
        
        new_sol2 = copy.deepcopy(sol2)
        new_sol2.fitness = None        
        
        variables = self.get_variables(sol1) 
        num_parameters = len(variables)
        num_parameters_changing = random.choice(rangeInt(1, int(num_parameters * Config.get_max_prec_crossover_replace_params_values())))
        variables_changing = random.sample(variables, num_parameters_changing)
        
        for variable in variables_changing:   
            setattr(new_sol2.item, variable, getattr(sol1.item, variable))
            setattr(new_sol1.item, variable, getattr(sol2.item, variable))
        
        return new_sol1, new_sol2
    
    def random(self, type):
        switcher = {
            SolutionType.DEEP_LEARNING: ItemDeepLearning.generate_item,
            SolutionType.RANDOM_FOREST: ItemRandomForest.generate_item,
            SolutionType.GRADIENT_BOOSTING_MACHINE: ItemGradientBoostingMachine.generate_item,
        }
        
        return Solution(switcher.get(type, None)())
    
    def reproduce(self, pop):  
        reproduced = {}
        for solution_type in SolutionType:
            reproduced[solution_type] = self.reproduceType(pop[solution_type])
        return reproduced
        
    def reproduceType(self, pop):
        mutateList,crossoverList, randomList  = divide_list(pop, Config.get_mutation_rate(), Config.get_crossover_rate())
        
        new_pop = []
        new_pop.extend(pop)
        
        for sol in mutateList:
            new_pop.append(self.mutate(sol))
            
        for i in range(0,len(crossoverList),2):
            c1, c2 = self.crossover(crossoverList[i], crossoverList[i+1])
            new_pop.append(c1)
            new_pop.append(c2)            
        
        for sol in randomList:
            new_pop.append(self.random(sol.getType()))
            
        return new_pop

    def selectType(self, pop):
        # Calculate the selection size based on the population size
        selection_size = math.ceil(len(pop) / 2)
        # Calculate the size of the random selection as % of the selection size
        random_selection_size = int(round(selection_size * Config.get_select_random_percent()))
        random_size = 0

        # Sort the population based on fitness in descending order
        sorted_population = list(set(pop))
        sorted_population.sort(key=lambda solution: solution.fitness, reverse=True)

        # Adjust the selection sizes if the sorted population is smaller
        if len(sorted_population) < selection_size:
            random_selection_size = 0
            # Calculate the number of random individuals needed to fill the selection
            random_size = selection_size - len(sorted_population)
            selection_size = len(sorted_population)

        # Select the top individuals based on fitness
        new_population = sorted_population[:selection_size - random_selection_size]

        # Select the remaining individuals randomly
        random_positions = random.sample(range(selection_size - random_selection_size, len(sorted_population)), k=random_selection_size)
        new_population.extend(sorted_population[i] for i in random_positions)

        # Add random individuals if necessary
        new_population.extend(self.random(pop[0].getType()) for i in range(random_size))

        return new_population
    
    def select(self, pop):
        selected = {}
        for solution_type in SolutionType:
            selected[solution_type] = self.selectType(pop[solution_type])
        return selected
    
    def run(self):  
        hist_path=Path(f'hist_{self.MachineLearning.id}.log', 'logs')      
        # initialization
        popModels = self.init_pop()
        
        #Population containing the populations with all the models (DL, RF, GBM)
        pop=[]
        historicPop = []

        for i in range(Config.get_max_gen()):
            print(f'Generation {i}')
            historicPop.append({})
            
            pop=[]
            
            for solution_type in SolutionType:                  
                # compute fitness of current population
                self.fitness_pop(popModels[solution_type], i)
                pop.extend(copy.deepcopy(popModels[solution_type]))
                historicPop[i][solution_type]=copy.deepcopy(popModels[solution_type])
                SaveLogHist(historicPop, hist_path)

                if Config.get_verbose():
                    print(f'\n======= {solution_type} || Generation {str(i)} =====\n')
                    self.print_pop(popModels[solution_type].__str__())
                    print('\n\nBest Fitness found: '+str(popModels[solution_type][0].fitness))
                    
            # select the best individuals to reproduce
            selected = self.select(popModels)
            # reproduce them
            popModels = self.reproduce(selected)

            if Config.get_verbose():
                print(f'\n======= All Algorithms Population || Generation {str(i)} =====\n')
                self.print_pop(pop)
                print('\n\nBest Fitness found: '+str(pop[0].fitness))
            
            self.MachineLearning.h2o_remove_all()
        
        self.MachineLearning.h2o_close()
        
        charts = Charts(historicPop)
        
        SaveLog(pop, historicPop, self.MachineLearning.id)
        
        fig_fitness_chart, _ = charts.generate_best_fitness_evolution_chart()
        savePlot(fig_fitness_chart, f'{self.MachineLearning.id}_Best_fitness_evolution_with_all_models')

        fig_fitness_chart1, _,  = charts.generate_best_fitness_evolution_by_model_chart()
        savePlot(fig_fitness_chart1, f'{self.MachineLearning.id}Fitness_evolution_by_model')
                
        return charts, historicPop
