import pandas as pd

with open('../input/split_20.txt', 'r') as r:
    run = []
    training = []
    validation = []
    testing = []
    for line in r:
        if line.startswith('RUN'):
            run.append(line.split(' ')[-1])
        elif line.startswith('Training'):
            training.append(' '.join(line.strip().split(':')[-1][2:-1].split(', ')))
        elif line.startswith('Validation'):
            validation.append(line.strip().split(' ')[-1])
        elif line.startswith('Test'):
            testing.append(line.strip().split(' ')[-1])

pd.DataFrame({'Run':run,
              'Training': training,
              'Validation': validation,
              'Test': testing}).to_csv('../output/splits.tsv', sep = '\t')