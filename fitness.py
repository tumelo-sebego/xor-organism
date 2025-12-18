import os
import neat

# 1. Define the inputs and expected outputs for XOR
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

def eval_genomes(genomes, config):
    """
    This function acts as the 'Teacher'. It looks at every genome (AI) 
    in the population and gives it a fitness score.
    """
    for genome_id, genome in genomes:
        genome.fitness = 4.0  # Start with a perfect score and subtract error
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            # Subtract the squared error from the fitness
            genome.fitness -= (output[0] - xo[0]) ** 2

def run_neat(config_file):
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population (the starting group of random networks)
    p = neat.Population(config)

    # Add a reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))

    # Run for up to 300 generations
    winner = p.run(eval_genomes, 300)

    # Show the final result
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run_neat(config_path)