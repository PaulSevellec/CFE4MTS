
import numpy as np
from models import ToyClassifier
from inceptionModel import InceptionModel
import torch

from sklearn.metrics import confusion_matrix


def train_classifier(train_dataset, valid_dataset, n_classes, n_var, n_timesteps, n_epochs, batch_size, device, dataset_name, model_type='toy'):

    if model_type == 'toy':
        model = ToyClassifier(n_var*n_timesteps, n_classes)
    elif model_type == 'Inception':
        model = InceptionModel(n_var, n_classes, 32, depth=6)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = 1000000000000000
    early_stopping = 0

    for epoch in range(n_epochs):
        model.train()
        for i, (x_batch, y_batch) in enumerate(train_dataset):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            y_pred = model(x_batch)

            # Compute Loss
            loss = criterion(y_pred, y_batch.long())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Validation
        model.eval()
        loss_total = 0
        accuracy_total = 0
        y_tot = []
        y_pred_tot = []
        with torch.no_grad():
            for x_batch, y_batch in valid_dataset:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch.long())
                loss_total += loss.item()
                y_classes = torch.argmax(y_pred, dim=1)
                accuracy = torch.sum(y_classes == y_batch).item() / len(y_batch)
                accuracy_total += accuracy
                y_tot.extend(y_batch.cpu().numpy())
                y_pred_tot.extend(y_classes.cpu().numpy())
        print(f'Epoch {epoch+1}')
        print(f'Validation loss: {loss_total/len(valid_dataset)}')
        print(f'Validation accuracy: {accuracy_total/len(valid_dataset)}')
        
        #Confusion matrix
        y_tot = np.vstack(y_tot)
        y_pred_tot = np.vstack(y_pred_tot)
        print('Confusion matrix')
        print(confusion_matrix(y_tot, y_pred_tot))
        
        if loss_total < best_loss:
            best_loss = loss_total
            torch.save(model.state_dict(), f'models/{dataset_name}/{model_type}_classifier.pth')
            print(f'Model saved at models/{dataset_name}/{model_type}_classifier.pth')
        else:
            early_stopping += 1
            if early_stopping == 5:
                print('Early stopping')
                break

    return model

def load_classifier(valid_dataset, n_var, n_timesteps, n_classes, dataset_name, device, model_type='toy'):
    
    if model_type == 'toy':
        model = ToyClassifier(n_var*n_timesteps, n_classes)
    elif model_type == 'Inception':
        model = InceptionModel(n_var, n_classes, 32, depth=6)
        
    model.load_state_dict(torch.load(f'models/{dataset_name}/{model_type}_classifier.pth', map_location=device))
    model.to(device)

    #Validation
    model.eval()
    accuracy_total = 0
    y_tot = []
    y_pred_tot = []
    with torch.no_grad():
        for x_batch, y_batch in valid_dataset:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            y_classes = torch.argmax(y_pred, dim=1)
            accuracy = torch.sum(y_classes == y_batch).item() / len(y_batch)
            accuracy_total += accuracy
            y_tot.extend(y_batch.cpu().numpy())
            y_pred_tot.extend(y_classes.cpu().numpy())

    print(f'Validation accuracy: {accuracy_total/len(valid_dataset)}')

    #Confusion matrix
    y_tot = np.vstack(y_tot)
    y_pred_tot = np.vstack(y_pred_tot)
    print('Confusion matrix')
    print(confusion_matrix(y_tot, y_pred_tot))

    return model


