import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from Python.model_data import model_data
from Python.rnn_data import VisitSequenceWithLabelDataset, seq_collate_fn
from Python.rnn_model import RNN, train, evaluate

def rnn_fit(params):
    scalers = []
    models = []
    auc = []
    verbose = params['verbose']
    gru_input = params['gru_input']
    hidden_dim = params['hidden_dim']
    layer_size = params['layer_size']
    dropout = params['dropout']
    for fold, (train_index, valid_index) in enumerate(skf.split(static_train.index, static_train.label)):
        y_train, y_valid = static_train.loc[static_train.index[train_index], ['label']], static_train.loc[static_train.index[valid_index], ['label']]

        x_train = y_train.join(seq_train).drop(columns=['label'])
        scaler = MinMaxScaler()
        scaled_x_train = scaler.fit_transform(x_train)
        scaled_x_train = pd.DataFrame(scaled_x_train, index=x_train.index, columns=x_train.columns)
        x_valid = y_valid.join(seq_train).drop(columns=['label'])
        scaled_x_valid = scaler.transform(x_valid)
        scaled_x_valid = pd.DataFrame(scaled_x_valid, index=x_valid.index, columns=x_valid.columns)
        train_dataset = VisitSequenceWithLabelDataset(scaled_x_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=seq_collate_fn, num_workers=NUM_WORKERS)
        valid_dataset = VisitSequenceWithLabelDataset(scaled_x_valid, y_valid)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=seq_collate_fn, num_workers=NUM_WORKERS)

        model = RNN(x_train.shape[1], gru_input, hidden_dim, layer_size, dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=.2)

        model.to(device)
        criterion.to(device)

        best_val_auc = 0.0
        train_losses, train_aucs = [], []
        valid_losses, valid_aucs = [], []
        early_stopping = 0
        if verbose > 0: print(f'===> Training model for fold {fold}')
        for epoch in range(NUM_EPOCHS):
            train_loss, train_auc = train(model, device, train_loader, criterion, optimizer, epoch)
            valid_loss, valid_auc = evaluate(model, device, valid_loader, criterion)
            scheduler.step(-valid_auc)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            train_aucs.append(train_auc)
            valid_aucs.append(valid_auc)
            
            if verbose > 0:
                print(f'Epoch: [{epoch}]\t'
                    'Train\t'
                    f'Loss {train_loss:.4f}\t'
                    f'AUC {train_auc:.4f}\t'
                    'Valid\t'
                    f'Loss {valid_loss:.4f}\t'
                    f'AUC {valid_auc:.4f}')
            
            is_best = valid_auc > best_val_auc
            if is_best:
                best_val_auc = valid_auc
                best_model = model
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping >= 10:
                    # print(f"early stopping at epoch {epoch}")
                    break

        scalers.append(scaler)
        models.append(best_model)
        auc.append(best_val_auc)

    if verbose != -1:
        print(params, np.mean(auc))
    return {
        'loss': -np.mean(auc),
        'scalers': scalers,
        'models': models,
        'params': params,
        'status': STATUS_OK,
    }

if __name__=='__main__':
    NUM_EPOCHS = 60
    BATCH_SIZE = 32
    USE_CUDA = True
    NUM_WORKERS = 8
    DATAPATH = "Data"
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('===> Loading datasets')

    static_train, static_test, seq_train, seq_test = model_data("RNN", test_size=.1)
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    print('===> Hyperparameter tuning')
    rnn_params = {
        'gru_input': hp.choice('gru_input', [16, 32, 64]), 
        'hidden_dim': hp.choice('hidden_dim', [16, 32, 64]), 
        'layer_size': hp.choice('layer_size', [2, 3, 4, 5]), 
        'dropout': hp.choice('dropout', [.1, .2, .3, .4, .5]),
        'verbose': 0
    }
    trials = Trials()
    rnn_best = fmin(rnn_fit, rnn_params, algo=tpe.suggest, rstate=np.random.default_rng(42), max_evals=50, trials=trials)

    print('===> Best model:')
    print(trials.best_trial['result']['models'][0])

    print('===> Save model')
    for fold, (scaler, model) in enumerate(zip(trials.best_trial['result']['scalers'], trials.best_trial['result']['models'])):
        pickle.dump(scaler, open(f'output/rnn_scaler{fold}.pkl', 'wb'))
        torch.save(model, os.path.join("./output", f"rnn_model{fold}.pth"), _use_new_zipfile_serialization=False)

    print('===> Evaluate test data')
    y_test = static_test[['label']]
    y_pred_cv = np.empty((y_test.size, 5))
    for fold in range(5):
        scaler = pickle.load(open(f'output/rnn_scaler{fold}.pkl', 'rb'))
        model = torch.load(os.path.join("./output", f"rnn_model{fold}.pth"))
        scaled_x_test = scaler.transform(seq_test)
        scaled_x_test = pd.DataFrame(scaled_x_test, index=seq_test.index, columns=seq_test.columns)
        test_dataset = VisitSequenceWithLabelDataset(scaled_x_test, y_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=seq_collate_fn, num_workers=NUM_WORKERS)

        model.eval()
        results = []
        
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):

                if isinstance(input, tuple):
                    input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
                else:
                    input = input.to(device)
                target = target.to(device)

                output = model(input)

                y_true = target.detach().to('cpu').numpy().tolist()
                y_pred = nn.Softmax(1)(output).detach().to('cpu').numpy()[:,1].tolist()
                results.extend(list(zip(y_true, y_pred)))
            
            y_true, y_pred_cv[:, fold] = zip(*results)
            auc = roc_auc_score(y_true, y_pred_cv[:, fold])

    print(f"AUC on test data: {roc_auc_score(y_true, y_pred_cv.mean(1))}")