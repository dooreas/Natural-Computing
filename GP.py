import copy
import inspect
import operator
import random
import math
from statistics import mean
from operator import attrgetter
import itertools
import matplotlib.pyplot as plt 
import numpy as np


#-*- coding: latin1 -*-

class Individual:
    gene = list()
    fitness = float('inf')
    results = list()

    def __init__(self):
        self.gene = list()
        self.fitness = float('inf')
        self.results = list()

    def update_fitness(self):
        global best_fitness
        global best_individual
        # self.gene = clean_individual(self.gene)
        self.results, self.fitness = rate_individual(self, self.gene)
        if self.fitness < best_fitness:
            best_fitness = self.fitness
            best_individual = self

def generate_individual():
    individual = Individual()
    for j in range(starting_individual_length):
        individual.gene.append(random.choice(random.choice(f + t)))
    i = 0
    while i < len(individual.gene) and individual.gene[i] in chained_f:
        del individual.gene[i]
        i+=1
    individual.update_fitness()
    return individual


def generate_population():
    _population = list()
    for i in range(population_size):
        _population.append(generate_individual())
    return _population


def clean_individual(_gene):
    trash_count = 1
    i = len(_gene)
    while i > 0 and trash_count > 0:
        i -= 1
        if _gene[i] in chained_f:
            trash_count += a[_gene[i]] - 1
        elif _gene[i] in t:
            trash_count -= 1
    _gene = _gene[i:]
    return _gene


def rate_individual(parent, _gene):
    fitness = float('inf')
    results = list()
    for y in y_range:
        for x in x_range:
            stack = list()
            result = ""
            for element in _gene:
                stack.append(element)
                if element in chained_f:
                    stack.pop()
                    arity = a[element]
                    operation = ''
                    try:
                        # print('stack: ' + str(stack))
                        # print('arity: ' + str(arity))
                        if arity == 1:
                            operation = element + '(' + stack[-1] + ')'
                            stack.pop()
                        elif arity == 2:
                            operation = element + stack[-1]
                            operation = stack[-2] + operation
                            stack.pop()
                            stack.pop()
                        # print('op: ' + operation)
                        result = str(eval(operation))
                        # print('result: ' + result)
                        stack.append(result)
                    except (IndexError, ValueError, ZeroDivisionError, NameError, OverflowError):
                        break
            try:
                result = stack.pop()
                # print('result: ' + result)
                results.append(eval(result))
            except (IndexError, ValueError, ZeroDivisionError, NameError, OverflowError):
                results.append(float('inf'))
    fitness = (np.array((list(map(operator.pow,list(map(operator.sub, results, target_values)),itertools.repeat(2,len(results))))))**2).sum()
    return results, fitness


def crossover(parents, can_mutate):
    _offspring1 = Individual()
    _offspring2 = Individual()
    point = random.randint(0,len(parents[0].gene)-1)
    _offspring1.gene = parents[0].gene[:point]
    _offspring2.gene = parents[0].gene[point:]
    point = random.randint(0, len(parents[1].gene) - 1)
    _offspring1.gene += parents[1].gene[point:]
    _offspring2.gene = parents[1].gene[:point] + _offspring2.gene

    if len(_offspring1.gene) > max_individual_length:
        _offspring1.gene = _offspring1.gene[max_individual_length:]
    if len(_offspring2.gene) > max_individual_length:
        _offspring2.gene = _offspring2.gene[max_individual_length:]

    if can_mutate:
        if (random.random() <= mutation_rate):
            _offspring1 = mutation(_offspring1)

        if (random.random() <= mutation_rate):
            _offspring2 = mutation(_offspring2)

    _offspring1.update_fitness()
    _offspring2.update_fitness()
    return (_offspring1, _offspring2)


def mutation(_individual):
    return crossover(list([_individual, generate_individual()]), False)[1]


def tournament(_population, _tournaments, _tournament_size):
    finals = list()
    for tournament in range(_tournaments):
        bracket = list()
        for individual in range(_tournament_size):
            bracket.append(random.choice(_population))
        bracket = sorted(bracket, key=attrgetter('fitness'))
        finals.append(bracket[0])
    return finals


def cull(_population):
    _population = sorted(_population, key=attrgetter('fitness'))
    return _population[:population_size]

if __name__ == "__main__":
    # initialize control vars
    global crossover_rate
    global mutation_rate
    # global permutation_rate
    global population_size
    global starting_individual_length
    global max_individual_length
    global max_generations
    global best_fitness
    global best_individual
    global error_threshold

    global inf
    inf = float('inf')

    crossover_rate = 0.95
    mutation_rate = 0.05
    # permutation_rate = 0.1
    best_fitness = float('inf')
    best_individual = Individual()
    error_threshold = 0.01

    global a
    global f
    global chained_f
    global t

    # YOU CAN MESS FROM HERE...

    population_size = 100
    starting_individual_length = 5
    max_individual_length = 5
    max_generations = 5000

    # terminal sets
    t = [['x'],
         ['1', "2", 'math.pi',"10"]]
    # function sets
    f = [['+', '-', '*'], # arithmetic
         # ['&', '|'], #bitwise
         [ 'math.cos'] # periodic
         ]
    # arity
    _a = [[2, 2, 2],
         # [2, 2],
         [1]]

    chained_f = list(itertools.chain.from_iterable(f))

    a = dict(zip(chained_f, (itertools.chain.from_iterable(_a))))

    # define optimization target
    objective_function = "10-10*math.cos(2*math.pi*x)+x**2"
    # objective_function = "(x + 2 * y - 7)**2 + (2 * x + y - 5)**2"

    # objective_function = "math.sqrt(x) + x**2"
    # objective_function = "math.sin(x) + math.log(x**2)"
    # objective_function = "math.log(x)"
    # objective_function = "math.sqrt(x)"
    # objective_function = "3*x*x*x + 7*x*x - x - 9"
    # objective_function = "2*x*x+3*x+4"
    # x_range = [x*0.1 for x in range(1, 101)]
    x_range = np.array([-5+x*0.1 for x in range(100)])
    y_range = range(0,1)

    # ...TO HERE

    global target_values
    target_values = list()
    for y in y_range:
        for x in x_range:
            target_values.append(eval(objective_function))

    # initialize population
    population = generate_population()

    generation = 1
    while generation < max_generations and best_fitness > error_threshold:
        if(random.random() <= crossover_rate):
            population += crossover(tournament(population, 2, 3), True)
            population = cull(population)
        # if(random.random() <= permutation_rate):
            # not yet implemented
            # pass
        # print(best_fitness)
        generation += 1
        print(best_fitness)
    # population = cull(population)
    #print("All fitness: " + str(list(element.fitness for element in population)))
    #print("Last generation: " + str(generation))
    print("Best fitness: " + str(best_individual.fitness))
    print("Best individual: " + str(best_individual.gene))
    print("Best individual length: " + str(len(best_individual.gene)))
    # print("Best operation: " + str(population[0].result))
    print("Expected value: " + str(target_values))
    print("Final value: " + str(best_individual.results))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.plot(np.array(x_range),np.array(target_values), "-r",label="Rastrigin")
    ax.plot(np.array(x_range),np.array(best_individual.results), "g.",label=str(best_individual.gene))
    
    ax.set_xlabel('x-value')
    ax.set_ylabel('y-value')


    plt.legend(loc="best")
    fig.tight_layout()
    fig,ax
    plt.show()