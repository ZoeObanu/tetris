import numpy as np
import pandas as pd
import random
from game import Game
import pickle
from custom_model import CUSTOM_AI_MODEL

def cross(a1, a2):
    # Crossover two agents (mix their weights based on fitness proportion)
    new_weights = []
    prop = a1.fit_rel / a2.fit_rel  # Proportional selection based on fitness
    
    # Combine weights from both parents based on some proportion
    for i in range(len(a1.weights)):
        rand = random.uniform(0, 1)
        if rand > prop:
            new_weights.append(a1.weights[i])
        else:
            new_weights.append(a2.weights[i])
    
    # Create and return the new agent with crossover weights
    new_weights = np.array(new_weights)  # Ensure weights are in the correct format (NumPy array)
    return CUSTOM_AI_MODEL(weights=new_weights, mutate=True)

def compute_fitness(agent, trials):
    rfitness = []
    pfitness = []
    
    for i in range(trials):
        game = Game('student', agent=agent)  # Initialize Game with the agent
        pieces_dropped, rows_cleared = game.run_no_visual()  # Run the game
        rfitness.append(rows_cleared)  # Store how well the agent did
        pfitness.append(pieces_dropped)  # Store how well the agent did
        print(f"trail {i+1} complete: rows cleared {rows_cleared}, pieces droped {pieces_dropped}")
    
    return np.average(np.array(pfitness))#, np.average(np.array(rfitness))  # Return average fitness

def run_X_epochs(num_epochs=5, num_trials=5, pop_size=50, num_elite=5, survival_rate=0.2):
    fitness_history = []
    top_fitness_history = []

    data = []

    # data collection over epochs
    data=[[1, np.ones(9), 1, np.ones(9),  1, np.ones(9)]]
    headers = ['avg_fit','avg_gene', 'top_fit', 'top_gene', 'elite_fit', 'elite_gene']
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(f'src/data/custom.csv', index=False)



    # Initialize population with random weights (e.g., 3 parameters for each agent)
    population = [CUSTOM_AI_MODEL(weights=np.random.randn(3)) for k in range(pop_size)]

    for epoch in range(num_epochs):
        total_fitness = 0
        #top_agent = None
        weights = np.zeros(3)

        # Evaluate fitness of all agents
        for n in range(pop_size):
            agent = population[n]
            agent.fit_score = compute_fitness(agent, num_trials)  # Calculate agent's fitness
            total_fitness += agent.fit_score
            weights += agent.weights

            print(f"agent {n+1} complete")

        # Calculate average fitness for the population
        avg_fitness = total_fitness / pop_size
        fitness_history.append(avg_fitness)
        # Get the top agent's fitness
        top_agent = sorted(population, key=lambda x: x.fit_score, reverse=True)[0]
        top_fitness_history.append(top_agent.fit_score)

        # Print the top agent's fitness at each epoch
        print(f"Epoch {epoch+1}: Avg Fitness: {avg_fitness:.2f}")
        print(f"Epoch {epoch+1}: Top Fitness: {top_agent.fit_score:.2f}\n")

        # Check if the model is improving
        if epoch > 0:
            improvement = avg_fitness - fitness_history[epoch - 1]
            print(f"Improvement from last epoch: {improvement:.2f}")
            timprovement = top_agent.fit_score - top_fitness_history[epoch - 1]
            print(f"Improvement from last epoch: {timprovement:.2f} \n\n")
        


        # Normalize relative fitness of each agent
        for agent in population:
            agent.fit_rel = agent.fit_score / total_fitness
        
        # Selection: sort the population by fitness (descending order)
        sorted_pop = sorted(population, key=lambda x: x.fit_score, reverse=True)

        # Elitism: Copy top-performing agents directly into the next generation
        elite_fit_score = 0
        elite_weights = np.zeros(3)
        next_gen = []

        for i in range(num_elite):
            elite_fit_score += sorted_pop[i].fit_score
            elite_weights += sorted_pop[i].weights
            next_gen.append(CUSTOM_AI_MODEL(weights=sorted_pop[i].weights, mutate=False))  # Elite don't mutate

        # Select parents for crossover (using the top portion of the population)
        num_parents = round(pop_size * survival_rate)
        parents = sorted_pop[:num_parents]

        # Create offspring through crossover
        for k in range(pop_size - num_elite):
            parent1, parent2 = random.sample(parents, 2)
            next_gen.append(cross(parent1, parent2))

        # Update the population for the next generation
        population = next_gen
        




        avg_fit = (total_fitness/pop_size)
        avg_gene = (weights/pop_size)
        top_fit = (top_agent.fit_score)
        top_gene = (top_agent.weights)
        elite_fit = (elite_fit_score/num_elite)
        elite_gene = (elite_weights/num_elite)

        data = [[avg_fit, avg_gene, top_fit, top_gene, elite_fit, elite_gene]]
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(f'src/data/custom.csv', mode='a', index=False, header=False)

    
    return data


run_X_epochs(num_epochs=5, num_trials=5, pop_size=50, num_elite=5, survival_rate=0.2)

