from __future__ import absolute_import
from __future__ import print_function

from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results
from sklearn.preprocessing import StandardScaler
from sklearn.impute import  SimpleImputer as Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import os
import numpy as np
import pandas as pd
import argparse
import json
import pickle

def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])

def AUROC(preds, dtrain):
    y = dtrain.get_label()
    AUROC = roc_auc_score(y, preds)
    return 'AUROC', AUROC

def binary_cross_entropy(pred, dtrain):
    true_label = dtrain.get_label()

    def get_single_model_label(y_proba_file):
        y_proba_path = os.path.join(os.getcwd(), '../teachers_predictions/{}/y_train_soft_labels.pkl'.format(y_proba_file))

        with open(y_proba_path, 'rb') as f:
            return pickle.load(f).squeeze()

    def get_GRU_mean_fusion_label(dim_depth_dict):
        y_proba_list = []
        for dim in dim_depth_dict['dim_list']:
            for depth in dim_depth_dict['depth_list']:
                y_proba_path = os.path.join(os.getcwd(), './teachers_predictions/gru_depth{}_dim{}/y_train_soft_labels.pkl'.format(depth, dim))

                with open(y_proba_path, 'rb') as f:
                    y_proba = pickle.load(f)
                y_proba_list.append(y_proba)
        return np.array(y_proba_list).mean(0).squeeze()

    def get_LSTM_mean_fusion_label(dim_depth_dict):
        y_proba_list = []
        for dim in dim_depth_dict['dim_list']:
            for depth in dim_depth_dict['depth_list']:

                y_proba_path = os.path.join(os.getcwd(), './teachers_predictions/lstm_depth{}_dim{}/y_train_soft_labels.pkl'.format(depth, dim))

                with open(y_proba_path, 'rb') as f:
                    y_proba = pickle.load(f)
                y_proba_list.append(y_proba)
        return np.array(y_proba_list).mean(0).squeeze()

    def get_all_mean_fusion_label(dim_depth_dict):
        y_proba_list = []
        for model in ['lstm', 'gru']:
            for dim in dim_depth_dict['dim_list']:
                for depth in dim_depth_dict['depth_list']:
                    y_proba_path = os.path.join(os.getcwd(), './teachers_predictions/{}_depth{}_dim{}/y_train_soft_labels.pkl'.format(model, depth, dim))

                    with open(y_proba_path, 'rb') as f:
                        y_proba = pickle.load(f)
                    y_proba_list.append(y_proba)
        return np.array(y_proba_list).mean(0).squeeze()

    dim_depth_dict = {
        'dim_list': [16, 32, 64],
        'depth_list': [2, 3]
    }

    if label_fusion == 'single_model':
        label = get_single_model_label(y_proba_file)
    elif label_fusion == 'GRU_mean_fusion':
        label = get_GRU_mean_fusion_label(dim_depth_dict)
    elif label_fusion == 'LSTM_mean_fusion':
        label = get_LSTM_mean_fusion_label(dim_depth_dict)
    elif label_fusion == 'all_mean_fusion':
        label = get_all_mean_fusion_label(dim_depth_dict)


    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    # sigmoid_pred = np.exp(pred / T) / (1 + np.exp(pred / T))
    label_logits = np.log(label / (1 - label))
    label = np.exp(label_logits / T) / (1 + np.exp(label_logits / T))

    # soft grad & hess
    soft_grad = -(1 ** label) * (label - sigmoid_pred)
    soft_hess = (1 ** label) * sigmoid_pred * (1.0 - sigmoid_pred)

    # hard grad & hess
    hard_grad = -(1 ** true_label) * (true_label - sigmoid_pred)
    hard_hess = (1 ** true_label) * sigmoid_pred * (1.0 - sigmoid_pred)

    grad = (1 - alpha) * soft_grad + alpha * hard_grad
    hess = (1 - alpha) * soft_hess + alpha * hard_hess

    return grad, hess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    parser.add_argument('--load_data_from_pickle', type=bool, default=True)
    parser.add_argument('--label_fusion', type=str, default='GRU_mean_fusion',
                        help='The way of fusing the labels from deep learning models')
    parser.add_argument('--y_proba_file', type=str, default='',
                        help='Type the y_proba_file name of the model to use, this can only be used when label_fusion is single mode')
    parser.add_argument('--Temperature', type=float, default=0.6,
                        help='Type the temperature of the distilling process.')
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='Type the weight of hard labels.')
    args = parser.parse_args()

    print(args)

    # Build data reader, discretizers, normalizers
    if not args.load_data_from_pickle:

        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                                 listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                 period_length=48.0)

        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                               listfile=os.path.join(args.data, 'val_listfile.csv'),
                                               period_length=48.0)

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                                listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                period_length=48.0)

        print('Reading data and extracting features ...')
        (train_X, train_y, train_names) = read_and_extract_features(train_reader, args.period, args.features)
        (val_X, val_y, val_names) = read_and_extract_features(val_reader, args.period, args.features)
        (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
        print('  train data shape = {}'.format(train_X.shape))
        print('  validation data shape = {}'.format(val_X.shape))
        print('  test data shape = {}'.format(test_X.shape))

        dataset_dict = {
            'train': {'train_X': train_X, 'train_y': train_y, 'train_names': train_names},
            'val': {'val_X': val_X, 'val_y': val_y, 'val_names': val_names},
            'test': {'test_X': test_X, 'test_y': test_y, 'test_names': test_names},
        }

        with open(os.path.join(args.output_dir, 'load_data', 'dataset_dict.pkl'), 'wb') as f:
            pickle.dump(dataset_dict, f)

    elif args.load_data_from_pickle:
        with open(os.path.join(args.output_dir, 'load_data', 'dataset_dict.pkl'), 'rb') as f:
            dataset_dict = pickle.load(f)

        train_X, train_y = dataset_dict['train']['train_X'], dataset_dict['train']['train_y']
        val_X, val_y = dataset_dict['val']['val_X'], dataset_dict['val']['val_y']
        test_X, test_y = dataset_dict['test']['test_X'], dataset_dict['test']['test_y']
        test_names = dataset_dict['test']['test_names']

    print('Imputing missing values ...')
    imputer = Imputer(missing_values=np.nan, strategy='mean', verbose=0, copy=True)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    print('Normalizing the data to have zero mean and unit variance ...')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    d_train = xgb.DMatrix(train_X, train_y)
    d_val = xgb.DMatrix(val_X, val_y)
    d_test = xgb.DMatrix(test_X, test_y)


    params = {
        'silent': 1,
        'objective': 'binary:logistic',
        'gamma': 0.1,
        'min_child_weight': 5,
        'disable_default_eval_metric': 1,
    }

    global label_fusion
    label_fusion = args.label_fusion
    global y_proba_file
    y_proba_file = args.y_proba_file
    global T
    T = args.Temperature
    global alpha
    alpha = args.alpha

    model = xgb.train(params=params,
                      dtrain=d_train,
                      num_boost_round=250,
                      feval=AUROC,
                      evals=[(d_train, 'dtrain'), (d_val, 'dval')],
                      obj=binary_cross_entropy)
  


    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    file_name = 'XKD_{}_{}_T{}_a{}'.format(label_fusion, y_proba_file, T, alpha)

    with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(train_y, model.predict(d_train))
        ret = {k : float(v) for k, v in ret.items()}
        json.dump(ret, res_file)


    with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(val_y, model.predict(d_val))
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    prediction = model.predict(d_test)

    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(test_y, prediction)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    save_results(test_names, prediction, test_y,
                 os.path.join(args.output_dir, 'predictions', file_name + '.csv'))

    val_prediction = pd.DataFrame.from_dict({'val_y': d_val.get_label(), 'val_y_proba': model.predict(d_val)})

    with open(os.path.join(args.output_dir, 'predictions', 'val_prediction_{}.pkl'.format(file_name)), 'wb') as f:
        pickle.dump(val_prediction, f)



if __name__ == '__main__':
    main()



