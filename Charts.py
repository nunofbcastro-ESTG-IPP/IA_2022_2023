import copy
import random
import statistics

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from Solution import Solution, SolutionType


class Charts:
    def __init__(self, historic_pop):
        self.historic_pop = copy.deepcopy(historic_pop)

    def generate_best_fitness_evolution_chart(self):
        df = pd.DataFrame(
            columns=['generation', 'type', 'configuration', 'best_fitness'])
        for i in range(len(self.historic_pop)):
            best_fitness = 0
            best_fitness_pop = []
            for solution_type in SolutionType:
                pop = self.historic_pop[i][solution_type]
                pop.sort(key=lambda sol: sol.fitness, reverse=True)
                if pop[0].fitness > best_fitness:
                    best_fitness_pop = pop

            row = pd.DataFrame({'generation': [i+1], 'configuration': [
                               best_fitness_pop[0].item.__str__()], 'best_fitness': [best_fitness_pop[0].fitness]})
            df = pd.concat([df, row])

        fig, ax = plt.subplots()
        ax.plot(df.generation, df.best_fitness, color='blue', marker='o')
        ax.set_title("Best fitness evolution with all models", fontsize=16)
        ax.set_xlabel("generation", fontsize=14)
        ax.set_ylabel("best fitness", color="blue", fontsize=14)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        return fig, df

    def generate_best_fitness_evolution_by_model_chart(self):
        df = pd.DataFrame(
            columns=['generation', 'type', 'configuration', 'best_fitness', 'average_fitness'])
        for i in range(len(self.historic_pop)):
            for solution_type in SolutionType:
                pop = self.historic_pop[i][solution_type]
                pop.sort(key=lambda sol: sol.fitness, reverse=True)
                row = pd.DataFrame({'generation': [i+1], 'type': [pop[0].getType().value], 'configuration': [pop[0].item.__str__(
                )], 'best_fitness': [pop[0].fitness], 'average_fitness': [statistics.mean(individual.fitness for individual in pop)]})
                df = pd.concat([df, row])

        fig, ax = plt.subplots()
        ax.set_title("Fitness evolution by model", fontsize=16)
        ax.set_xlabel("generation", fontsize=14)
        ax.set_ylabel("best fitness", color="blue", fontsize=14)

        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

        legend_labels = []

        for solution_type in SolutionType:
            filtered_df = df[df['type'] == solution_type.value]
            random_color = random.choice(colors)
            colors.remove(random_color)
            ax.plot(filtered_df.generation, filtered_df.best_fitness,
                    color=random_color, marker='o')
            legend_labels.append(solution_type.value)

        ax.legend(legend_labels, loc='lower center', bbox_to_anchor=(
            0.5, -0.25), ncol=len(legend_labels), fontsize=12)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.subplots_adjust(bottom=0.2)

        fig.set_figheight(7)
        fig.set_figwidth(8)

        return fig, df

    @staticmethod
    def generate_model_scoring_history_chart(model, metric):
        scoring_history = model.scoring_history()
        fig, ax = plt.subplots()

        if hasattr(scoring_history, 'epochs'):
            xAxis = scoring_history.epochs
            xAxisName = 'EPOCHS'
        else:
            xAxis = scoring_history.number_of_trees
            xAxisName = 'NRTREES'
        
        if metric == 'rmse':
            print('A gerar com rmse')
            yAxisTraining = scoring_history.training_rmse
            yAxisValidation = scoring_history.validation_rmse
        else:
            print('A gerar com logloss')
            yAxisTraining = scoring_history.training_logloss
            yAxisValidation = scoring_history.validation_logloss
            
        metric = metric.upper()
        ax.plot(xAxis, yAxisTraining, c='blue', label=f'Training {metric}')
        ax.plot(xAxis, yAxisValidation, c='orange', label=f'Validation {metric}')
        
        ax.set_title(f'Scoring History - {metric}')
        ax.set_xlabel(xAxisName)
        ax.set_ylabel(metric)
        ax.legend()

        return fig
