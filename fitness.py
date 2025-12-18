import os
import pickle
import neat

# 1. Define the XOR logic
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

def run_neat(config_path):
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create population and run evolution
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    winner = p.run(eval_genomes, 300)

    # --- SAVING THE WINNER ---
    print(f'\nSaving winner to winner.pkl...')
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    return winner

def load_and_test_winner(config_path):
    """How to use the saved organism later"""
    # 1. Load the genome back from the file
    with open('winner.pkl', 'rb') as f:
        genome = pickle.load(f)

    # 2. Re-load the config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # 3. Create the 'Brain' (Phenotype)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # 4. Test it
    print("\n--- Testing Loaded Organism ---")
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        print(f"Input: {xi} | Expected: {xo[0]} | AI Output: {output[0]:.4f}")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    
    # Run evolution and save
    run_neat(config_path)
    
    # Demonstrate loading the file we just created
    load_and_test_winner(config_path)