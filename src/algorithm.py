import gc
import copy
import torch
import numpy as np

from joblib import Parallel, delayed
from functools import partial

from .utils import initiate_model, get_accuracy, CalibrationError, set_lambda



###############################
# Update models in the client #
###############################
# FedAvg, LG-FedAvg, FedPer
def basic_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # prepare model
    model.train()
    model = initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)

    # prepare optimizer       
    optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9)

    # main loop
    for e in range(args.E if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            # get loss
            losses += loss.item()
            
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        if (args.device == 'cuda') and (torch.cuda.device_count() > 1):
            model = model.module.to('cpu')
        else:
            model = model.to('cpu')
    return model



# FedProx
def fedprox_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # set local model
    local_model = copy.deepcopy(model)
    local_model.train()
    local_model = initiate_model(local_model, args)
    
    # set global model for a regularization
    previous_optimal_model = copy.deepcopy(model)
    for parameter in previous_optimal_model.parameters(): parameter.requires_grad = False
    del model; gc.collect()

    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)

    # prepare optimizer       
    optimizer = optimizer(local_model.parameters(), lr=lr, momentum=0.9)
    
    # main loop
    for e in range(args.E if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = local_model(inputs)
            loss = criterion()(outputs, targets)
            
            # calculate proximity regularization
            prox = 0.
            for p_local, p_global in zip(local_model.parameters(), previous_optimal_model.parameters()): 
                prox += (p_local - p_global.to(args.device)).norm(2)
            loss += args.mu * prox
            
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            # get loss
            losses += loss.item()
            
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del previous_optimal_model; gc.collect()
        if (args.device == 'cuda') and (torch.cuda.device_count() > 1):
            model = local_model.module.to('cpu')
        else:
            model = local_model.to('cpu')
    return local_model



# FedRep
def fedrep_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    assert args.tau > 0, '[ERROR] argument `tau` should be properly assigned!'
    
    # prepare model
    model.train()
    model = initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)

    # update head (penultimate layer) first
    for name, parameter in model.named_parameters():
        if 'classifier' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer       
    head_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if 'classifier' in name], 
        lr=lr, 
        momentum=0.9
    )
    
    # head fine-tuning loop
    for e in range(args.tau if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)

            # update
            head_optimizer.zero_grad()
            loss.backward()
            head_optimizer.step() 
            
            # get loss
            losses += loss.item()
            
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [HEAD EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del head_optimizer; gc.collect()
        
    # then, update the body
    for name, parameter in model.named_parameters():
        if 'classifier' in name:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True
    
    # prepare optimizer 
    body_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if 'classifier' not in name], 
        lr=lr, 
        momentum=0.9
    )
    
    # body adaptation loop
    for e in range(args.E if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)

            # update
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step() 
            
            # get loss
            losses += loss.item()
            
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [BODY EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del body_optimizer; gc.collect()
        
        if (args.device == 'cuda') and (torch.cuda.device_count() > 1):
            model = model.module.to('cpu')
        else:
            model = model.to('cpu')
    return model



# APFL
def apfl_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # get current personalized model based on the previous global omdel
    personalized_model = copy.deepcopy(model)
    personalized_model.apply(partial(set_lambda, lam=args.apfl_constant))
    
    # prepare model
    personalized_model.train()
    personalized_model = initiate_model(personalized_model, args)
    
    # update local model only
    for name, parameter in personalized_model.named_parameters():
        if '_local' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False

    # prepare optimizer       
    local_optimizer = optimizer(
        [parameter for name, parameter in personalized_model.named_parameters() if '_local' in name], 
        lr=lr, 
        momentum=0.9
    )
    
    # wait, get the global model, too
    model.apply(partial(set_lambda, lam=0.0))

    # prepare model
    model.train()
    model = initiate_model(model, args)
    
    # update global model only
    for name, parameter in model.named_parameters():
        if '_local' in name:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True
    
    # prepare optimizer       
    global_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if '_local' not in name], 
        lr=lr, 
        momentum=0.9
    )
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)
    
    # update global model & personalized model
    for e in range(args.E if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs); local_outputs = personalized_model(inputs)
            global_loss = criterion()(outputs, targets); local_loss = criterion()(local_outputs, targets)

            # update
            global_optimizer.zero_grad(); local_optimizer.zero_grad()
            global_loss.backward(); local_loss.backward()
            global_optimizer.step(); local_optimizer.step()
            
            # get loss
            losses += global_loss.item() + local_loss.item()
            
            # temporarliy merge into personalized model for metrics
            with torch.no_grad():
                temp_model = copy.deepcopy(model)
                local_updated = {k: v for k, v in personalized_model.state_dict().items() if '_local' in k}
                global_updated = model.state_dict()
                global_updated.update(local_updated)
                temp_model.load_state_dict(global_updated)
                
                temp_model.apply(partial(set_lambda, lam=args.apfl_constant))
                outputs = temp_model(inputs)
                del temp_model; gc.collect()
                
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del global_optimizer, local_optimizer; gc.collect()
        
        # merge updated local model with updated global model 
        local_updated = {k: v for k, v in personalized_model.state_dict().items() if '_local' in k}
        global_updated = model.state_dict()
        global_updated.update(local_updated)
        model.load_state_dict(global_updated)
        del personalized_model, local_updated, global_updated; gc.collect()
        
        if (args.device == 'cuda') and (torch.cuda.device_count() > 1):
            model = model.module.to('cpu')
        else:
            model = model.to('cpu')
    return model



# Ditto
def ditto_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    assert args.tau > 0, '[ERROR] argument `tau` should be properly assigned!'
    
    # prepare model
    model.train()
    model = initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)
    
    # get the global model first
    model.apply(partial(set_lambda, lam=0.0))
    
    # update global model only
    for name, parameter in model.named_parameters():
        if '_local' in name:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True
    
    # prepare optimizer       
    global_optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9)

    # update global model 
    for e in range(args.E if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            global_loss = criterion()(outputs, targets)

            # update
            global_optimizer.zero_grad()
            global_loss.backward()
            global_optimizer.step() 
            
            # get loss
            losses += global_loss.item()
            
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [GLOBAL MODEL EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del global_optimizer; gc.collect()
        
    # then, get the local model
    model.apply(partial(set_lambda, lam=1.0))
    
    # update local model only
    for name, parameter in model.named_parameters():
        if '_local' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer       
    local_optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9)
    
    # then update local model
    for e in range(args.tau if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            local_loss = criterion()(outputs, targets)
            
            # calculate regularization term toward optimal global model
            prox = 0.
            weights = model.state_dict()
            for name in weights.keys():
                if '_local' not in name:
                    continue
                prox += (weights[name[:-6]] - weights[name]).norm(2)
            else:
                del weights; gc.collect()
            local_loss += args.mu * prox
            
            # update
            local_optimizer.zero_grad()
            local_loss.backward()
            local_optimizer.step() 
            
            # get loss
            losses += local_loss.item()
            
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [LOCAL MODEL EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del local_optimizer; gc.collect()
        
        if (args.device == 'cuda') and (torch.cuda.device_count() > 1):
            model = model.module.to('cpu')
        else:
            model = model.to('cpu')
    return model



# pFedMe
def pfedme_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # prepare model
    model.train()
    model = initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)
    
    # get the local model first
    model.apply(partial(set_lambda, lam=1.0))

    # update global model only
    for name, parameter in model.named_parameters():
        if '_local' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer       
    optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9)
    
    # update local model first
    for e in range(args.E if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)
            
            # calculate regularization term toward global model
            prox = 0.
            weights = model.state_dict()
            for name in weights.keys():
                if '_local' not in name:
                    continue
                prox += (weights[name[:-6]] - weights[name]).norm(2)
            else:
                del weights; gc.collect()
            loss += args.mu * prox
            
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            # get loss
            losses += loss.item()
            
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:
            # update global model based on updated local model weights (line 8 of Algorihtm 1 in the paper)
            weights = model.state_dict()
            for name in weights.keys():
                if '_local' not in name:
                    continue
                weights[name[:-6]] -= lr * args.mu * (weights[name[:-6]] - weights[name])
            else:
                model.load_state_dict(weights)
                del weights; gc.collect()
            
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        if (args.device == 'cuda') and (torch.cuda.device_count() > 1):
            model = model.module.to('cpu')
        else:
            model = model.to('cpu')
    return model

    
    
    
# SuPerFed-MM (Model-wise Mixture) & SuPerFed-LM (Layer-wise Mixture)
def superfed_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs, start_mix):            
    # retrieve mode (MM or LM)
    mode = args.algorithm[-2:]
    
    # set global model for a regularization
    if args.mu > 0:
        previous_optimal_model = copy.deepcopy(model)
        for parameter in previous_optimal_model.parameters(): parameter.requires_grad = False
    
    # prepare model
    model.train()
    model = initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)

    # prepare optimizer       
    optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9)
    
    # update local model first
    for e in range(args.E if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            if start_mix:
                # instant model mixing according to mode
                if mode == 'mm':
                    lam = np.random.uniform(0.0, 1.0)
                    model.apply(partial(set_lambda, lam=lam))
                elif mode == 'lm':
                    model.apply(partial(set_lambda, lam=None, layerwise=True))
            else:
                # update global model before setting `start_mixning` flag
                model.apply(partial(set_lambda, lam=0.0))
       
            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)
            
            # calculate proximity regularization term toward global model
            if args.mu > 0:
                prox = 0.
                for p_local, p_global in zip(model.parameters(), previous_optimal_model.parameters()): 
                    prox += (p_local - p_global.to(args.device)).norm(2)
                loss += args.mu * prox
                
            # subspace construction
            if start_mix:
                weights = model.state_dict()
                numerator, norm_1, norm_2 = 0, 0, 0
                for name in weights.keys():
                    if '_local' not in name:
                        continue
                    numerator += (weights[name[:-6]] * weights[name]).sum()
                    norm_1 += weights[name[:-6]].pow(2).sum()
                    norm_2 += weights[name].pow(2).sum()
                else:
                    del weights; gc.collect()
                    cos_sim = numerator.pow(2) / (norm_1 * norm_2)
                loss += args.nu * cos_sim
            
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            # get loss
            losses += loss.item()
            
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:            
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        if (args.device == 'cuda') and (torch.cuda.device_count() > 1):
            model = model.module.to('cpu')
        else:
            model = model.to('cpu')
    return model


    
###################
# Evaluate models #
###################
@torch.no_grad()
def basic_evaluate(identifier, args, model, criterion, dataset):
    # prepare model
    model.eval()
    model.to(args.device)
    
    if args.algorithm in ['apfl', 'ditto', 'pfedme', 'superfed-mm', 'superfed-lm']:
        # get the personalized or local model
        model.apply(partial(set_lambda, lam=args.apfl_constant if args.algorithm == 'apfl' else 1.0))

    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=False, pin_memory=True)
    
    # track loss and metrics
    losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
    
    # main loop
    for inputs, targets in dataloader:
        # get daata
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        
        # inference
        outputs = model(inputs)

        # get loss
        losses += criterion()(outputs, targets).item()

        # get accuracy
        accs = get_accuracy(outputs, targets, (1, 5))
        acc1 += accs[0].item(); acc5 += accs[-1].item()

        # get calibration errors
        ces = CalibrationError()(outputs, targets)
        ece += ces[0].item(); mce += ces[-1].item()

        # clear cache
        if args.device == 'cuda': torch.cuda.empty_cache()
    else:
        losses /= len(dataloader)
        acc1 /= len(dataloader)
        acc5 /= len(dataloader)
        ece /= len(dataloader)
        mce /= len(dataloader)
        if identifier is not None:
            print(f'\t[EVALUATION - CLIENT ({str(identifier).zfill(4)})] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
        else:
            print(f'\t[EVALUATION - SERVER] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    return losses, acc1, acc5, ece, mce



@torch.no_grad()
def superfed_evaluate(identifier, args, model, criterion, dataset, current_round):
    results = []
    for lam in np.arange(0.0, 1.1, 0.1):
        # prepare model
        model.eval()
        model.to(args.device)
        
        # make dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=False, pin_memory=True)
    
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0

        # main loop
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # model mixing with given value
            model.apply(partial(set_lambda, lam=lam))

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)

            # get loss
            losses += loss.item()

            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if args.device == 'cuda': torch.cuda.empty_cache()
        else:
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            results.append([losses, acc1, acc5, ece, mce])
    
    # get best metrics to print
    results = torch.tensor(results).T
    lowest_loss_idx = results[0].argmin()
    losses, acc1, acc5, ece, mce = results[:, lowest_loss_idx].squeeze()
    

    print(f'\t[EVALUATION - CLIENT ({str(identifier).zfill(4)})] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    return results
