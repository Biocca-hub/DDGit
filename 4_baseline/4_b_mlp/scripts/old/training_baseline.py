# IMPORTARE FUNZIONE PER CROSS VALIDATION SUI 20 SPLIT RIGOROSI
from model_definition import cv_run as c

# DEFINIZIONE IPERPARAMETRI PER CROSS VALIDATION
folder = '../output'
hidden_dims = [16,16,16]
pat = 50
warm = 150
max_ep = 1000
output = 'out.txt'
scatter = 'scat.txt'

# TRAINING
c(hidden_dims, pat, max_ep, warm, output, scatter, folder)