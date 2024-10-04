import torch
import torch.nn as nn
import torch.nn.functional as F
from models import LSTMDiscriminator, CLSTMDiscriminator, Discriminator, Generator, LSTMGenerator, CMLPGenerator, MLPGenerator, BiLSTMGenerator, CBiLSTMGenerator
import os
import numpy as np 
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import IsolationForest
from collections import namedtuple
import datetime
#Needed to allow computation of the gradients for noisers 
torch.backends.cudnn.enabled=False

def GAN_train(dataset_name, train_dataset, valid_dataset, scaler, classifier, n_classes, n_var, n_timesteps, n_epochs, n_batch, batch_size, G, D, CD_type, with_CD, with_LD, with_LN, criterion, device, lambda_1, lambda_2, lambda_3, lambda_4):

    alpha = 0.1
    generator_lr = 0.001
    discriminator_lr = 0.001
    central_discriminator_lr = 0.0001

    id_temps = np.array(range(n_timesteps))
    id_temps = torch.Tensor(id_temps).to(device)

    discriminators = {}
    if D == 'LSTM':
        cond_D = False
        for i in range(n_var):
            discriminators[i] = LSTMDiscriminator(ts_dim=n_timesteps)
    elif D == 'CLSTM':
        cond_D = True
        for i in range(n_var):
            discriminators[i] = CLSTMDiscriminator(ts_dim=n_timesteps+n_classes)
    elif D == 'MLP':
        cond_D = False
        for i in range(n_var):
            discriminators[i] = Discriminator(n_samples=n_timesteps, alpha=alpha)

    for i in range(n_var):
        discriminators[i] = nn.DataParallel(discriminators[i]).to(device)
        discriminators[i].to(device)  
 
    if with_LN:
        noisers = {}
        if G == 'LSTM':
            cond_G = False
            for i in range(n_var):
                noisers[i] = LSTMGenerator(latent_dim=n_timesteps+n_classes, ts_dim=n_timesteps)
        elif G == 'BiLSTM':
            cond_G = False
            for i in range(n_var):
                noisers[i] = BiLSTMGenerator(latent_dim=n_timesteps, ts_dim=n_timesteps)
        elif G == 'CBiLSTM':
            cond_G = True
            for i in range(n_var):
                noisers[i] = CBiLSTMGenerator(latent_dim=n_timesteps+n_classes, ts_dim=n_timesteps)
        elif G == 'MLP':
            cond_G = False
            for i in range(n_var):
                noisers[i] = Generator(noise_len=n_timesteps+n_classes, n_samples=n_timesteps, alpha=alpha)
        for i in range(n_var):
            noisers[i] = nn.DataParallel(noisers[i])
            noisers[i].to(device)
    else:
        
        if D == 'LSTM':
            G = 'MLP'
            cond_G = False
            noiser = MLPGenerator(n_timesteps, 0.3, n_var = n_var, shrink=True)
        elif D == 'CLSTM':
            G = 'CMLP'
            cond_G = True
            noiser = CMLPGenerator(n_timesteps, 0.3, n_var = n_var, shrink=True)
        noiser = nn.DataParallel(noiser)
        noiser.to(device)
        print('Noiser: MLP_CFE4SITS')

    if criterion == 'BCE':
        loss_function = nn.BCELoss()
    elif criterion == 'MSE':
        loss_function = nn.MSELoss()

    start_time = datetime.datetime.now()

    try:
        if not os.path.isdir(f'./models/{dataset_name}/{start_time.month}_{start_time.day}_{start_time.hour}_{start_time.minute}_{G}_{D}_{CD_type}_{with_CD}_{with_LD}_{with_LN}_{n_epochs}_{lambda_1}_{lambda_2}_{lambda_3}_{lambda_4}'):
            os.mkdir(f'./models/{dataset_name}/{start_time.month}_{start_time.day}_{start_time.hour}_{start_time.minute}_{G}_{D}_{CD_type}_{with_CD}_{with_LD}_{with_LN}_{n_epochs}_{lambda_1}_{lambda_2}_{lambda_3}_{lambda_4}')
    except:  # noqa: E722
        pass

    dir_path = f'./models/{dataset_name}/{start_time.month}_{start_time.day}_{start_time.hour}_{start_time.minute}_{G}_{D}_{CD_type}_{with_CD}_{with_LD}_{with_LN}_{n_epochs}_{lambda_1}_{lambda_2}_{lambda_3}_{lambda_4}'
    print('Directory path :', dir_path)

    mse = nn.MSELoss()

    optimizers_D = {}
    for i in range(n_var):
        optimizers_D[i] = torch.optim.Adam(discriminators[i].parameters(), lr=discriminator_lr, betas=[0.5, 0.9])

    if with_LN:
        optimizers_G = {}
        for i in range(n_var):
            optimizers_G[i] = torch.optim.Adam(noisers[i].parameters(), lr=generator_lr, betas=[0.5, 0.9])
    else:
        optimizers_G = torch.optim.Adam(noiser.parameters(), lr=generator_lr, betas=[0.5, 0.9])

    if with_CD:
        if CD_type == 'LSTM':
            cond_cd = False
            central_discriminator = LSTMDiscriminator(ts_dim=n_timesteps*n_var, num_layers=n_var)
        if CD_type == 'CLSTM':
            cond_cd = True
            central_discriminator = CLSTMDiscriminator(ts_dim=n_timesteps*n_var + n_classes , num_layers=n_var)
        elif CD_type == 'MLP':
            cond_cd = False
            central_discriminator = Discriminator(n_samples = n_var*n_timesteps, alpha = alpha)
            central_discriminator = central_discriminator
    
        central_discriminator = nn.DataParallel(central_discriminator)
        central_discriminator.to(device)
        optimizer_central_discriminator = torch.optim.Adam(central_discriminator.parameters(), lr=central_discriminator_lr, betas=[0.5, 0.9])
    else:
        cond_cd = False

    if with_LN:
        log_loss_G = np.zeros((n_epochs, n_var, 5))
    else:
        log_loss_G = np.zeros((n_epochs, 1, 5))
    log_loss_D = np.zeros((n_epochs, n_var))
    log_loss_CD = np.zeros((n_epochs))
    log_validity = np.zeros((n_epochs))

    classifier.eval()
    
    best_loss = 1000000000000000
    early_stopping = 0

    for epoch in range(n_epochs):

        cond_val_epoch = 0
        val_epoch = 0

        for i in range(n_var):
            if with_LN:
                noisers[i].train()
            if with_LD:
                discriminators[i].train()
        if with_CD:
            central_discriminator.train()
        if not with_LN:
            print('Noiser train')
            noiser.train()

        nb_batch = 0
        for x_batch, y_batch in train_dataset:
            
            nb_batch += 1
            #print(f'Batch {nb_batch}')

            batch_size = x_batch.shape[0]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = torch.argmax(classifier(x_batch), axis=1)
            random_labels = torch.randint(1, n_classes, (batch_size,)).to(device).long()
            random_labels = (random_labels+y_pred.long()) % n_classes
            y_ohe = F.one_hot(random_labels, num_classes=n_classes)

            # Generating noise
            if with_LN:
                generated_noise = {}
                for i in range(n_var):
                    generated_noise[i] = noisers[i](x_batch[:,i], y_ohe).float().detach()
                generated_noise_unified = torch.cat([generated_noise[i] for i in range(n_var)], dim=1)
                generated_noise_unified = generated_noise_unified.reshape(batch_size, n_var, n_timesteps)
            else:
                #print(x_batch.shape, y_ohe.shape)
                #print(noiser)
                generated_noise_unified = noiser(x_batch, y_ohe).float().detach()


            generated_samples_labels = torch.zeros((batch_size, 1)).to(device).float()
            real_samples_labels = torch.ones((batch_size, 1)).to(device).float()
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
            

            if with_LD:
                lambda_ld = 1.0
                # Data for training the discriminators
                all_samples_group = {}
                for i in range(n_var):
                    all_samples_group[i] = torch.cat(
                        (x_batch[:,i], generated_noise_unified[:,i])
                    )

                # Training the discriminators
                outputs_D = {}
                loss_D = {}
                for i in range(n_var):
                    optimizers_D[i].zero_grad()
                    outputs_D[i] = discriminators[i](all_samples_group[i].float(), torch.cat((F.one_hot(y_pred.long(), num_classes=n_classes),y_ohe)))
                    loss_D[i] = loss_function(outputs_D[i], all_samples_labels)
                    loss_D[i].backward()
                    optimizers_D[i].step()
                    log_loss_D[epoch, i] += loss_D[i].cpu().detach().numpy()
            else:
                lambda_ld = 0.0

            if with_CD:
                lambda_cd = 1.0
                if with_LN:
                    # Data from central discriminator
                    temp_generated = generated_noise_unified[:,0]
                    for i in range(1,n_var):
                        temp_generated = torch.hstack((temp_generated, generated_noise_unified[:,i]))
                    group_generated = temp_generated
                else:
                    group_generated = generated_noise_unified.reshape(batch_size, n_var*n_timesteps)

                temp_real = x_batch[:,0]
                for i in range(1,n_var):
                    temp_real = torch.hstack((temp_real, x_batch[:,i]))
                group_real = temp_real

                all_samples_central = torch.cat((group_generated, group_real))
                all_samples_labels_central = torch.cat(
                    (torch.zeros((len(x_batch), 1)).to(device).float(), torch.ones((len(x_batch), 1)).to(device).float())
                )

                # Training the central discriminator
                optimizer_central_discriminator.zero_grad()
                output_central_discriminator = central_discriminator(all_samples_central.float(), torch.cat((y_ohe, F.one_hot(y_pred.long(), num_classes=n_classes))))
                loss_central_discriminator = loss_function(
                    output_central_discriminator, all_samples_labels_central)
                loss_central_discriminator.backward()
                optimizer_central_discriminator.step()
                log_loss_CD[epoch] += loss_central_discriminator.cpu().detach().numpy()
            else:
                lambda_cd = 0.0

            # Training the generators
            outputs_G = {}
            loss_G_local = {}
            loss_G = {}

            if with_LN: 
                for i in range(n_var):
                    optimizers_G[i].zero_grad()

                    y_pred = torch.argmax(classifier(x_batch), axis=1)
                    y_ohe_pred = F.one_hot(y_pred.long(), num_classes=n_classes)
                    random_labels = torch.randint(1, n_classes, (batch_size,)).to(device).long()
                    random_labels = (random_labels+y_pred.long()) % n_classes
                    y_ohe = F.one_hot(random_labels, num_classes=n_classes)

                    # Generating CFs
                    generated_noise = {}
                    for j in range(n_var):
                        generated_noise[j] = noisers[i](x_batch[:,j], y_ohe).float()
                    generated_noise_unified = torch.cat([generated_noise[i] for i in range(n_var)], dim=1)
                    generated_noise_unified = generated_noise_unified.reshape(batch_size, n_var, n_timesteps)

                    if with_LD:
                        outputs_G[i] = discriminators[i](generated_noise_unified[:,i], y_ohe)
                        loss_G_local[i] = loss_function(outputs_G[i], real_samples_labels)
                    else:
                        loss_G_local[i] = 0

                    # Compute MSE between the generated CFs and the original data
                    #uni_reg = mse(generated_noise_unified, x_batch)

                    to_add_abs = torch.abs(generated_noise_unified - x_batch)
                    _, t_avg = torch.max(to_add_abs,dim=-1,keepdim=True)
                    diff = torch.minimum(torch.remainder(t_avg - id_temps, n_timesteps),
                                        torch.remainder(id_temps - t_avg, n_timesteps))
                    weights = torch.square(diff+1)
                    uni_reg = torch.sum( weights * to_add_abs) / batch_size / n_var / n_timesteps

                    if cond_G or cond_D:   
                        prob_cf = classifier(generated_noise_unified)
                        prob_cf = torch.nn.functional.softmax(prob_cf,dim=1)
                        prob = torch.sum(prob_cf * y_ohe,dim=1)
                        loss_classif = torch.mean( -torch.log( prob + torch.finfo(torch.float32).eps ) )
                    else:
                        prob_cf = classifier(generated_noise_unified)   
                        prob_cf = torch.nn.functional.softmax(prob_cf,dim=1)
                        prob = torch.sum(prob_cf * y_ohe_pred,dim=1)
                        loss_classif = torch.mean( -torch.log( 1. - prob + torch.finfo(torch.float32).eps ) )

                    loss_central_discriminator_new = 0
                    if with_CD:
                        output_central_discriminator_new = {}

                        temp_generated = generated_noise_unified[:,0]
                        for j in range(1,n_var):
                            temp_generated = torch.hstack((temp_generated, generated_noise_unified[:,j]))
                        group_generated = temp_generated

                        output_central_discriminator_new[i] = central_discriminator(group_generated, y_ohe) 
                        loss_central_discriminator_new = loss_function(output_central_discriminator_new[i], real_samples_labels)
                        
                    loss_G[i] = lambda_ld * lambda_1 * loss_G_local[i] + lambda_cd * lambda_2 * loss_central_discriminator_new + lambda_3 * loss_classif + lambda_4 * uni_reg

                    cond_val_epoch += torch.where(prob_cf.argmax(dim=1) == random_labels, 1, 0).sum().item()
                    val_epoch += torch.where(prob_cf.argmax(dim=1) != y_pred, 1, 0).sum().item()
                    
                    #print('LD', lambda_ld*lambda_1*loss_G_local[i].detach().cpu().numpy(), 'CD', lambda_cd * lambda_2 * loss_central_discriminator_new.detach().cpu().numpy(), 'classif', lambda_3 * loss_classif.detach().cpu().numpy(), 'uni', lambda_4 * uni_reg.detach().cpu().numpy())
                    #print('Loss G', loss_G[i].detach().cpu().numpy(), 'loss G loc', lambda_ld*lambda_1*loss_G_local[i].detach().cpu().numpy(), 'loss cd', lambda_cd * lambda_2 * loss_central_discriminator_new.detach().cpu().numpy(), 'loss classif', lambda_3 * loss_classif.detach().cpu().numpy(), 'loss uni', lambda_4 * uni_reg.detach().cpu().numpy())

                    log_loss_G[epoch, i, 0] += loss_G[i].detach().cpu().numpy()
                    if with_LD:
                        log_loss_G[epoch, i, 1] += lambda_1 * loss_G_local[i].detach().cpu().numpy()
                    else:
                        log_loss_G[epoch, i, 1] = 0
                    
                    if with_CD:
                        log_loss_G[epoch, i, 2] += lambda_2 * loss_central_discriminator_new.detach().cpu().numpy()
                    else:
                        log_loss_G[epoch, i, 2] = 0
                    log_loss_G[epoch, i, 3] += lambda_3 * loss_classif.detach().cpu().numpy()
                    log_loss_G[epoch, i, 4] += lambda_4 * uni_reg.detach().cpu().numpy()

                    loss_G[i].backward()
                    optimizers_G[i].step()
                    
                    loss_G_local[i] = 0
                    loss_G[i] = 0
                    loss_classif = 0
                    uni_reg = 0
                    loss_central_discriminator_new = 0
                    #print("Batch Loss G : ", loss_G[i].detach().cpu().numpy(), " - Loss D : ", loss_D[i].detach().cpu().numpy(), " - Loss CD : ", loss_central_discriminator_new[i].detach().cpu().numpy(), " - Loss classif : ", loss_classif.detach().cpu().numpy(), " - Loss uni : ", uni_reg.detach().cpu().numpy())
            else:
                optimizers_G.zero_grad()

                # Generating CFs
                generated_noise_unified = noiser(x_batch, y_ohe).float()

                loss_G_local = 0
                if with_LD:
                    for i in range(n_var):
                        outputs_G = discriminators[i](generated_noise_unified[:,i], y_ohe)
                        loss_G_local += loss_function(outputs_G, real_samples_labels)
                
                # Compute MSE between the generated CFs and the original data
                #uni_reg = mse(generated_noise_unified, x_batch)

                to_add_abs = torch.abs(generated_noise_unified - x_batch)
                _, t_avg = torch.max(to_add_abs,dim=-1,keepdim=True)
                diff = torch.minimum(torch.remainder(t_avg - id_temps, n_timesteps),
                                    torch.remainder(id_temps - t_avg, n_timesteps))
                weights = torch.square(diff+1)
                uni_reg = torch.sum( weights * to_add_abs) / batch_size / n_var / n_timesteps 

                if cond_G or cond_D:   
                    prob_cf = classifier(generated_noise_unified)
                    prob_cf = torch.nn.functional.softmax(prob_cf,dim=1)
                    prob = torch.sum(prob_cf * y_ohe,dim=1)
                    loss_classif = torch.mean( -torch.log( prob + torch.finfo(torch.float32).eps ) )
                else:
                    prob_cf = classifier(generated_noise_unified)   
                    prob_cf = torch.nn.functional.softmax(prob_cf,dim=1)
                    y_pred = torch.argmax(classifier(x_batch), axis=1)
                    y_ohe_pred = F.one_hot(y_pred.long(), num_classes=n_classes)
                    prob = torch.sum(prob_cf * y_ohe_pred,dim=1)
                    loss_classif = torch.mean( -torch.log( 1. - prob + torch.finfo(torch.float32).eps ) )

                loss_central_discriminator_new = 0
                if with_CD:
                    output_central_discriminator_new = {}
                    loss_central_discriminator_new = {}

                    temp_generated = generated_noise_unified[:,0]
                    for j in range(1,n_var):
                        temp_generated = torch.hstack((temp_generated, generated_noise_unified[:,j]))
                    group_generated = temp_generated
                    
                    output_central_discriminator_new = central_discriminator(group_generated, y_ohe) 
                    loss_central_discriminator_new = loss_function(output_central_discriminator_new, real_samples_labels)

                loss_G = lambda_ld * lambda_1 * loss_G_local + lambda_cd * lambda_2 * loss_central_discriminator_new + lambda_3 * loss_classif + lambda_4 * uni_reg

                cond_val_epoch += torch.where(prob_cf.argmax(dim=1) == random_labels, 1, 0).sum().item()
                val_epoch += torch.where(prob_cf.argmax(dim=1) != y_batch, 1, 0).sum().item()

                log_loss_G[epoch, 0, 0] += loss_G.detach().cpu().numpy()
                if with_LD:
                    log_loss_G[epoch, 0, 1] += lambda_1 * loss_G_local.detach().cpu().numpy()
                else:
                    log_loss_G[epoch, 0, 1] = 0
                    
                if with_CD:
                    log_loss_G[epoch, 0, 2] += lambda_2 * loss_central_discriminator_new.detach().cpu().numpy()
                else:
                    log_loss_G[epoch, 0, 2] = 0
                log_loss_G[epoch, 0, 3] += lambda_3 * loss_classif.detach().cpu().numpy()
                log_loss_G[epoch, 0, 4] += lambda_4 * uni_reg.detach().cpu().numpy()

                loss_G.backward()
                optimizers_G.step()

                loss_G_local = 0
                loss_G = 0
                loss_classif = 0
                uni_reg = 0
                loss_central_discriminator_new = 0

        log_loss_G[epoch] /= n_batch
        log_loss_D[epoch] /= n_batch
        log_loss_CD[epoch] /= n_batch

        print(f'Epoch {epoch} - Done - Mean loss G : {np.mean(log_loss_G[epoch, :, 0])} - D : {np.mean(log_loss_D[epoch])} - CD : {log_loss_CD[epoch]} - G local : {np.mean(log_loss_G[epoch, :, 1])} - G CD : {np.mean(log_loss_G[epoch, :, 2])} - G classif : {np.mean(log_loss_G[epoch, :, 3])} - G uni : {np.mean(log_loss_G[epoch, :, 4])}')
        print(f'Nb of valid CFs : {val_epoch} - Nb of cond valid CFs : {cond_val_epoch}')

        # Validation pass 
        nb_cf_valid = 0
        nb_conditional_cf_valid = 0
        nb_cf_generated = 0
        generated_cf = []
        generated_cf_labels = []
        x_tot = []
        y_tot = []
        cond_labels = []

        loss_G = 0

        for i in range(n_var):
            if with_LN:
                noisers[i].eval()
            if with_LD:
                discriminators[i].eval()
        if with_CD:
            central_discriminator.eval()
        if not with_LN:
            noiser.eval()

        for x_valid, y_valid in valid_dataset:
            batch_size = x_valid.shape[0]
            x_valid_numpy = x_valid.numpy()
            x_valid = x_valid.to(device)

            #x_transform = minirocket.transform(x_valid_numpy)
            y_pred = classifier(x_valid)
            y_pred = torch.argmax(y_pred, axis=1)

            random_labels = torch.randint(1, n_classes, (batch_size,)).to(device).long()
            random_labels = (random_labels+y_pred.long()) % n_classes
            y_ohe = F.one_hot(random_labels, num_classes=n_classes)

            real_samples_labels_valid = torch.ones((batch_size, 1)).to(device).float()
            
            #t_1 = time()
            # Generating CFs
            if with_LN:
                generated_noise = {}
                for i in range(n_var):
                    generated_noise[i] = noisers[i](x_valid[:,i], y_ohe).float().detach()
                generated_noise_unified = torch.cat([generated_noise[i] for i in range(n_var)], dim=1)
                generated_noise_unified = generated_noise_unified.reshape(batch_size, n_var, n_timesteps)
            else:
                generated_noise_unified = noiser(x_valid, y_ohe).float().detach()
            #t_2 = time()
            #print((t_2-t_1)/batch_size)

            loss_G_local = 0
            if with_LD:
                for i in range(n_var):
                    outputs_G = discriminators[i](generated_noise_unified[:,i], y_ohe)
                    loss_G_local += loss_function(outputs_G, real_samples_labels_valid)
                
            # Compute MSE between the generated CFs and the original data
            uni_reg = mse(generated_noise_unified, x_valid)

            prob_cf = classifier(generated_noise_unified)
            prob = torch.sum(prob_cf * y_ohe,dim=1)
            loss_classif = torch.mean( -torch.log( prob + torch.finfo(torch.float32).eps ) )

            loss_central_discriminator_new = 0
            if with_CD:
                output_central_discriminator_new = {}
                loss_central_discriminator_new = {}

                temp_generated = generated_noise_unified[:,0]
                for j in range(1,n_var):
                    temp_generated = torch.hstack((temp_generated, generated_noise_unified[:,j]))
                group_generated = temp_generated 
                      
                output_central_discriminator_new_valid = central_discriminator(group_generated, y_ohe) 
                loss_central_discriminator_new = loss_function(output_central_discriminator_new_valid, real_samples_labels_valid)

            loss_G += (lambda_ld * lambda_1 * loss_G_local + lambda_cd * lambda_2 * loss_central_discriminator_new + lambda_3 * loss_classif + lambda_4 * uni_reg).item()

            x_cf_valid_numpy = generated_noise_unified.cpu().detach().numpy()
            if len(x_cf_valid_numpy) == batch_size:
                generated_cf.append(x_cf_valid_numpy)
                x_tot.append(x_valid_numpy)
                cond_labels.append(random_labels.cpu().detach().numpy())

            pred_cl = classifier(generated_noise_unified)

            if len(pred_cl) == batch_size:
                generated_cf_labels.append(torch.argmax(pred_cl, dim=1).cpu().detach().numpy())
                y_tot.append(y_pred.cpu().detach().numpy())

            nb_cf_valid += torch.sum(torch.argmax(pred_cl, dim=1) != y_pred).item()
            nb_conditional_cf_valid += torch.sum(torch.argmax(pred_cl, dim=1) == random_labels).item()
            nb_cf_generated += batch_size
            log_validity[epoch] = nb_cf_valid/nb_cf_generated*100

        generated_cf = np.vstack(generated_cf)
        generated_cf_labels = np.hstack(generated_cf_labels)
        x_valid_numpy = np.vstack(x_tot)
        y_pred_valid_numpy = np.hstack(y_tot)
        cond_labels = np.hstack(cond_labels)

        #print(generated_cf.shape, generated_cf_labels.shape, x_valid_numpy.shape, y_pred_valid_numpy.shape)

        print(f'Number of CF generated : {nb_cf_generated} - Number of valid CF {nb_cf_valid} - Percentage : {nb_cf_valid/nb_cf_generated*100}% - Loss G valid : {loss_G}')
        print('Confusion matrix')
        print(confusion_matrix(y_pred_valid_numpy, generated_cf_labels))
                    
        if best_loss > loss_G:
            best_loss = loss_G
            early_stopping = 0

            # Separate the generated CFs by their classes
            generated_cf_by_class = {}
            input_ts_by_class = {}
            generated_cf_labels_by_class = {}
            for i in range(n_classes):
                
                generated_cf_by_class[i] = generated_cf[y_pred_valid_numpy != i]
                generated_cf_by_class[i] = scaler.inverse_transform(generated_cf_by_class[i])
                input_ts_by_class[i] = x_valid_numpy[y_pred_valid_numpy != i]
                input_ts_by_class[i] = scaler.inverse_transform(input_ts_by_class[i])

                generated_cf_labels_by_class[i] = generated_cf_labels[y_pred_valid_numpy != i]
                
                #print(f'Class {i} - Number of CFs : {len(generated_cf_by_class[i])}, Number of input time series : {len(input_ts_by_class[i])} - Number of labels : {len(generated_cf_labels_by_class[i])}')
                #print shapes 
                #print(generated_cf_by_class[i].shape, input_ts_by_class[i].shape, generated_cf_labels_by_class[i].shape)
                
                #save data
                np.save(f'{dir_path}/generated_cf_{i}.npy', generated_cf_by_class[i])
                np.save(f'{dir_path}/input_ts_{i}.npy', input_ts_by_class[i])
                np.save(f'{dir_path}/generated_cf_labels_{i}.npy', generated_cf_labels_by_class[i])

            index_valid = np.where(y_pred_valid_numpy != generated_cf_labels)

            generated_cf_valid = generated_cf[index_valid]
            x_valid_numpy_valid = x_valid_numpy[index_valid]

            index_cond_valid = np.where(cond_labels == generated_cf_labels)

            generated_cf_cond_valid = generated_cf[index_cond_valid]
            x_valid_numpy_cond_valid = x_valid_numpy[index_cond_valid]

            save_metrics(dataset_name, G, D, CD_type, with_CD, with_LD, with_LN, epoch, lambda_1, lambda_2, lambda_3, lambda_4, x_valid_numpy, generated_cf, generated_cf_labels, y_pred_valid_numpy, generated_cf_valid, x_valid_numpy_valid, generated_cf_cond_valid, x_valid_numpy_cond_valid, scaler, nb_cf_valid, nb_conditional_cf_valid, nb_cf_generated, cond_G, cond_D, cond_cd, start_time)

        else:
            print('Early stopping')
            early_stopping += 1
            if early_stopping == 5:
                break
    
    #save_metrics(dataset_name, G, D, CD_type, with_CD, with_LD, with_LN, epoch, lambda_1, lambda_2, lambda_3, lambda_4, x_valid_numpy, generated_cf, generated_cf_labels, y_pred_valid_numpy, nb_cf_valid, nb_conditional_cf_valid, nb_cf_generated, cond_G, cond_D, cond_cd, start_time)
    print("Training done")

def save_metrics(dataset_name, G, D, CD_type, with_CD, with_LD, with_LN, epoch, lambda_1, lambda_2, lambda_3, lambda_4, x_valid_numpy, generated_cf, generated_cf_labels, y_valid_numpy, generated_cf_valid, x_valid_numpy_valid, generated_cf_cond_valid, x_valid_numpy_cond_valid, scaler, nb_cf_valid, nb_conditional_cf_valid, nb_cf_generated, cond_G, cond_D, cond_cd, start_time):   
    
    # Proximity
    proximity = compute_proximity(scaler.inverse_transform(x_valid_numpy), scaler.inverse_transform(generated_cf))
    if len(x_valid_numpy_valid) == 0:
        proximity_valid = 0
    else:
        proximity_valid = compute_proximity(scaler.inverse_transform(x_valid_numpy_valid), scaler.inverse_transform(generated_cf_valid))

    if len(x_valid_numpy_cond_valid) == 0:
        proximity_cond_valid = 0
    else:
        proximity_cond_valid = compute_proximity(scaler.inverse_transform(x_valid_numpy_cond_valid), scaler.inverse_transform(generated_cf_cond_valid))
    print(f"Proximity : {proximity}")
    print(f"Proximity valid : {proximity_valid}")
    print(f"Proximity cond valid : {proximity_cond_valid}")

    # Compacity
    threshold = 0.25
    compacity = compute_compacity(x_valid_numpy, generated_cf, threshold)
    if len(x_valid_numpy_valid) == 0:
        compacity_valid = 0
    else:
        compacity_valid = compute_compacity(x_valid_numpy_valid, generated_cf_valid, threshold)
    
    if len(x_valid_numpy_cond_valid) == 0:
        compacity_cond_valid = 0
    else:
        compacity_cond_valid = compute_compacity(x_valid_numpy_cond_valid, generated_cf_cond_valid, threshold)
    print(f"Compacity at threshold {threshold}: {compacity*100}%")
    print(f"Compacity valid at threshold {threshold}: {compacity_valid*100}%")
    print(f"Compacity cond valid at threshold {threshold}: {compacity_cond_valid*100}%")

    # Plausibility
    plausibility = compute_plausibility(x_valid_numpy, generated_cf)
    if len(x_valid_numpy_valid) == 0:
        plausibility_valid = 0
    else:
        plausibility_valid = compute_plausibility(x_valid_numpy_valid, generated_cf_valid)

    if len(x_valid_numpy_cond_valid) == 0:
        plausibility_cond_valid = 0
    else:
        plausibility_cond_valid = compute_plausibility(x_valid_numpy_cond_valid, generated_cf_cond_valid)
    print(f"Plausibility : {plausibility*100}%")
    print(f"Plausibility valid : {plausibility_valid*100}%")
    print(f"Plausibility cond valid : {plausibility_cond_valid*100}%")

    # Validity 
    print(f'Validity : {nb_cf_valid/nb_cf_generated*100}%')    

    # Conditional Validity
    print(f'Conditional Validity : {nb_conditional_cf_valid/nb_cf_generated*100}%')

    print(proximity)
    print(compacity*100)
    print(plausibility*100)
    print(nb_cf_valid/nb_cf_generated*100)
    print(nb_conditional_cf_valid/nb_cf_generated*100)
    print('\n')

    #time = datetime.datetime.now()

    # Save the results in a log file 
    with open(f'logs/{start_time.month}_{start_time.day}_{start_time.hour}_{start_time.minute}_{dataset_name}_{G}_{D}_{CD_type}_{with_CD}_{with_LD}_{with_LN}_{lambda_1}_{lambda_2}_{lambda_3}_{lambda_4}_log.txt', 'w') as f:
        f.write(f'G: {G}, with LN: {with_LN}\n')
        f.write(f'D: {D}, with LD: {with_LD}\n')
        f.write(f'CD: {CD_type}, with CD: {with_CD}\n')
        f.write(f'Epochs: {epoch}\n')
        f.write(f'Lambda 1: {lambda_1}, Lambda 2: {lambda_2}, Lambda 3: {lambda_3}, Lambda 4: {lambda_4}\n')
        f.write(f'Conditional G: {cond_G}, Conditional D: {cond_D}, Conditional CD: {cond_cd}\n')
        f.write(f'{nb_cf_valid/nb_cf_generated*100}%\n')
        f.write(f'{nb_conditional_cf_valid/nb_cf_generated*100}%\n')
        f.write(f"{compacity*100}%\n")
        f.write(f"{proximity}\n")
        f.write(f"{plausibility*100}%\n")
        f.write(np.array2string(confusion_matrix(y_valid_numpy, generated_cf_labels), precision=2, separator=','))
        f.write('\n')
        f.write(f"Only for valid CFs\n")
        f.write(f"{compacity_valid*100}%\n")
        f.write(f"{proximity_valid}\n")
        f.write(f"{plausibility_valid*100}%\n")
        f.write(f"Only for conditional valid CFs\n")
        f.write(f"{compacity_cond_valid*100}%\n")
        f.write(f"{proximity_cond_valid}\n")
        f.write(f"{plausibility_cond_valid*100}%\n")
        f.write(f"{nb_conditional_cf_valid/nb_cf_generated*100} & {compacity*100} & {proximity} & {plausibility*100} & {compacity_cond_valid*100} & {proximity_cond_valid} & {plausibility_cond_valid*100}\n")
        f.write(f"{nb_cf_valid/nb_cf_generated*100} & {compacity*100} & {proximity} & {plausibility*100} & {compacity_valid*100} & {proximity_valid} & {plausibility_valid*100}\n")

def compute_proximity(real, generated):
    proximity = np.linalg.norm(real - generated, axis=(1,2))
    return np.mean(proximity)

def compute_compacity(real, generated, threshold):
    compacity = np.abs(real-generated) <= threshold
    return np.mean(compacity)

def compute_plausibility(real, generated):
    real = real.reshape(real.shape[0], -1)
    generated = generated.reshape(generated.shape[0], -1)
    print(real.shape, generated.shape)
    estimator = IsolationForest().fit(real)
    ratios = plausibility_ratios(generated, estimator)
    return ratios.inlier

def _to_0_1(pred):
    """sklearn predicts in/outliers with values {+1,-1}
    This remaps them to {1,0}"""
    pred += 1
    pred[np.where(pred == 2)] = 1
    pred = pred.astype("int")
    return pred

def plausibility_ratios(X, outlier_estimator):
    ratiotup = namedtuple("ratiotup", ["inlier", "outlier", "total"])
    preds = _to_0_1(outlier_estimator.predict(X))
    n_outlier, n_inlier = np.bincount(preds, minlength=2)
    n_total = X.shape[0]
    ratios = ratiotup(
        inlier=n_inlier/n_total,
        outlier=n_outlier/n_total,
        total=n_total
    )
    return ratios