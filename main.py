import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from classifier_training import train_classifier, load_classifier
from models import StandardScaler
from GAN_training import GAN_train
import argparse
#from sklearn import preprocessing

DATA_DIR = "data/SpokenArabicDigits"

def main(train_classif, train_noiser, test_noiser, G_type, D_type, CD_type, with_CD, with_LD, with_LN, nb_epochs, batch_size, dataset_name, classifier_type, smoke_test, lambda1, lambda2, lambda3, lambda4):
    year = 2020

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    
    torch.manual_seed(0)
    print('\n=========\nManual seed activated for reproducibility\n=========')
    
    print(f"Working directory: {os.getcwd()}")

    x_train = np.load(f"{DATA_DIR}/x_train_{year}.npy")
    y_train = np.load(f"{DATA_DIR}/y_train_{year}.npy")
    x_valid = np.load(f"{DATA_DIR}/x_valid_{year}.npy")
    y_valid = np.load(f"{DATA_DIR}/y_valid_{year}.npy")
    x_test = np.load(f"{DATA_DIR}/x_test_{year}.npy")
    y_test = np.load(f"{DATA_DIR}/y_test_{year}.npy")
    
    if smoke_test:
        x_train = x_train[:2*batch_size]
        y_train = y_train[:2*batch_size]
        x_valid = x_valid[:2*batch_size]
        y_valid = y_valid[:2*batch_size]
        x_test = x_test[:2*batch_size]
        y_test = y_test[:2*batch_size]
        nb_epochs = 2
        batch_size = 5
        print("=====================")
        print("Smoke test activated")
        print("=====================")

    scaler = StandardScaler()
    
    # TRAIN DATA
    if dataset_name == 'SITS':
        x_train = x_train.transpose(0,2,1)
        y_train = y_train - 1
    #x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    #x_train = (x_train - 0.5)*2
    #x_train = x_train[:,0:5,:]
    x_train_tensor = torch.Tensor(x_train)
    x_train_tensor = scaler.fit_transform(x_train_tensor)
    x_train = x_train_tensor.numpy()
    print("X train tensor shape", x_train_tensor.shape, "Y train shape", y_train.shape)
    y_train_tensor = torch.Tensor(y_train)

    n_batch = int(len(x_train) / batch_size)

    # VALID DATA
    if dataset_name == 'SITS':
        x_valid = x_valid.transpose(0,2,1) 
        y_valid = y_valid - 1
    #x_valid = (x_valid - x_valid.min()) / (x_valid.max() - x_valid.min())
    #x_valid = (x_valid - 0.5)*2
    #x_valid = x_valid[:,0:5,:]
    x_valid_tensor = torch.Tensor(x_valid)
    x_valid_tensor = scaler.transform(x_valid_tensor)
    x_valid = x_valid_tensor.numpy()
    print("X valid tensor shape", x_valid_tensor.shape)
    y_valid_tensor = torch.Tensor(y_valid)

    # TEST DATA
    if dataset_name == 'SITS':
        x_test = x_test.transpose(0,2,1)
        y_test = y_test - 1
    #x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())
    #x_test = (x_test - 0.5)*2
    #x_test = x_test[:,0:5,:]
    x_test_tensor = torch.Tensor(x_test)
    x_test_tensor = scaler.transform(x_test_tensor)
    x_test = x_test_tensor.numpy()
    print("X test tensor shape", x_test_tensor.shape)
    y_test_tensor = torch.Tensor(y_test)

    print("Possible classes", y_train_tensor.unique())

    # Create dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor) 
    test_dataset  = TensorDataset(x_test_tensor, y_test_tensor) 
    valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor) 
    n_classes = len(np.unique(train_dataset.tensors[1]))
    if dataset_name == 'SITS':
        train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        valid_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    else :
        train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataset  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        valid_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    

    if train_classif :
        classifier = train_classifier(train_dataset, valid_dataset, n_classes, x_train_tensor.shape[1], x_train_tensor.shape[2], nb_epochs, batch_size, device, dataset_name, model_type=classifier_type)
        print("Classifier trained")
    else : 
        classifier = load_classifier(valid_dataset, x_train_tensor.shape[1], x_train_tensor.shape[2], n_classes, dataset_name, device, model_type=classifier_type)

    if train_noiser :
        GAN_train(dataset_name, train_dataset, valid_dataset, scaler, classifier, n_classes, x_train_tensor.shape[1], x_train_tensor.shape[2], nb_epochs, n_batch, batch_size, G_type, D_type, CD_type, with_CD, with_LD, with_LN, 'BCE', device, lambda1, lambda2, lambda3, lambda4)
        print("Noiser trained")

    #if test_noiser :
        #testNoiserMiniRocket(test_dataset, train_dataset, noiser_file_path, discr_file_path, classifier_file_path, classifier, minirocket, WGAN)
        #print("Noiser tested")   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--with_CD', type=str, default='False')
    parser.add_argument('--with_LD', type=str, default='False')
    parser.add_argument('--with_LN', type=str, default='False')
    parser.add_argument('--G', type=str, default='LSTM')
    parser.add_argument('--D', type=str, default='LSTM')
    parser.add_argument('--CD_type', type=str, default='MLP')
    parser.add_argument('--nb_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='SpokenArabicDigits')
    parser.add_argument('--smoke_test', type=str, default='False')
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--lambda3', type=float, default=1.0)
    parser.add_argument('--lambda4', type=float, default=0.1)
    parser.add_argument('--train_classif', type=str, default='False')
    parser.add_argument('--classifier_type', type=str, default='toy')

    with_CD = (parser.parse_args().with_CD == 'True')
    with_LD = (parser.parse_args().with_LD == 'True')
    with_LN = (parser.parse_args().with_LN == 'True')
    G_type = parser.parse_args().G
    D_type = parser.parse_args().D
    CD_type = parser.parse_args().CD_type
    nb_epochs = parser.parse_args().nb_epochs
    batch_size = parser.parse_args().batch_size
    dataset_name = parser.parse_args().dataset
    smoke_test = (parser.parse_args().smoke_test == 'True')
    lambda1 = parser.parse_args().lambda1
    lambda2 = parser.parse_args().lambda2
    lambda3 = parser.parse_args().lambda3
    lambda4 = parser.parse_args().lambda4
    train_classif = (parser.parse_args().train_classif == 'True')
    classifier_type = parser.parse_args().classifier_type

    print('G type', G_type, 'D type', D_type, 'CD type', CD_type, 'with CD', with_CD, 'with LD', with_LD, 'with LN', with_LN, 'nb epochs', nb_epochs, 'batch size', batch_size, 'dataset', dataset_name, 'smoke test', smoke_test, 'lambda1', lambda1, 'lambda2', lambda2, 'lambda3', lambda3, 'lambda4', lambda4, 'train classif', train_classif, 'classifier type', classifier_type)
    #print("With CD", with_CD, "G type", G_type, "D type", D_type, "CD type", CD_type, "nb epochs", nb_epochs, "batch size", batch_size, "dataset", dataset_name, "smoke test", smoke_test, "lambda1", lambda1, "lambda2", lambda2, "lambda3", lambda3, "lambda4", lambda4, "train classif", train_classif, "classifier type", classifier_type)

    train_noiser  = True
    test_noiser   = True


    if dataset_name == 'SpokenArabicDigits':
        DATA_DIR = "data/SpokenArabicDigits"
    elif dataset_name == 'SITS':
        DATA_DIR = "data/SITS"
    elif dataset_name == 'Synthetic':
        DATA_DIR = "data/Synthetic"
    elif dataset_name == 'ECG' :
        DATA_DIR = "data/ECG"

    main(train_classif, train_noiser, test_noiser, G_type, D_type, CD_type, with_CD, with_LD, with_LN, nb_epochs, batch_size, dataset_name, classifier_type, smoke_test, lambda1, lambda2, lambda3, lambda4)   
