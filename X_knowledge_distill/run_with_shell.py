import pandas as pd
import numpy as np

import re
import os


def run_single_model_label_fusion():
      label_fusion = 'single_model'
      Temperature = 1.0
      alpha = 0

      for model in ['gru', 'lstm']:
            for depth in [2, 3]:
                  for dim in [16, 32, 64]:
                        y_proba_file = '{}_depth{}_dim{}'.format(model, depth, dim)
                        cmd = 'python -um mimic3models.in_hospital_mortality.X_knowledge_distill.main --label_fusion {} ' \
                              '--y_proba_file {} ' \
                              '--Temperature {} ' \
                              '--alpha {}'.format(label_fusion, y_proba_file, Temperature, alpha)

                        os.system(cmd)

def run_ensemble_model_label_fusion():
    for label_fusion in ['LSTM_mean_fusion', 'GRU_mean_fusion']:
        for T in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
                cmd = 'python -um mimic3models.in_hospital_mortality.X_knowledge_distill.main ' \
                      '--label_fusion {}' \
                      ' --Temperature {} ' \
                      '--alpha {}'.format(label_fusion, T, alpha)
                os.system(cmd)

if __name__ == '__main__':
      # run_single_model_label_fusion()
      run_ensemble_model_label_fusion()
