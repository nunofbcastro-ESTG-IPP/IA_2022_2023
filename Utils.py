import random
import math
import numpy as np

def divide_number(number, n):
    quotient, remainder = divmod(number, n)
    if n <= 3:
        result = [quotient] * n
        result[:remainder] = [quotient + 1] * remainder
    else:
        result = [quotient] * n
        result[:remainder] = [x + 1 for x in result[:remainder]]
    return result[:n]

def divide_list(list, mutation_list_percent, crossover_list_percent):
    random.shuffle(list) 
    
    total_items = len(list)
    mutation_list_size = min(math.ceil(total_items * mutation_list_percent), total_items)
    crossover_list_size = min(math.ceil(total_items * crossover_list_percent), total_items-mutation_list_size)
    
    if crossover_list_size > 0 and crossover_list_size % 2 != 0:
        crossover_adjustment = random.choice([-1, 1])
        
        crossover_list_size = crossover_list_size + crossover_adjustment
        
        if random.choice([True, False]) == True:
            mutation_list_size = mutation_list_size - crossover_adjustment
    
    mutation_list = list[:mutation_list_size]
    crossover_list = list[mutation_list_size:mutation_list_size+crossover_list_size]
    random_list = list[mutation_list_size+crossover_list_size:]
    
    return mutation_list, crossover_list, random_list

def rangeInt(minValue: int, maxValue: int, step: int = 1):
    return range(minValue, maxValue+1, step)

def rangeFloat(minValue, maxValue, step=1):
    values = np.arange(minValue, maxValue+step, step)
    return values[values <= maxValue]