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
    initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)

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
            for param in model.parameters(): param.grad = None
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
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        model.to('cpu')



# FedProx
def fedprox_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # set local model
    model.train()
    initiate_model(model, args)
    
    # set global model for a regularization
    previous_global_model = copy.deepcopy(model)
    for parameter in previous_global_model.parameters(): parameter.requires_grad = False

    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)

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
            
            # calculate proximity regularization
            prox = 0.
            for p_local, p_global in zip(model.parameters(), previous_global_model.parameters()): 
                prox += (p_local - p_global.to(args.device)).norm(2)
            loss += args.mu * prox
            
            # update
            for param in model.parameters(): param.grad = None
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
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        model.to('cpu')



# FedRep
def fedrep_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    assert args.tau > 0, '[ERROR] argument `tau` should be properly assigned!'
    
    # prepare model
    model.train()
    initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)

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
            for param in model.parameters(): param.grad = None
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
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [HEAD EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del head_optimizer
        
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
            for param in model.parameters(): param.grad = None
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
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [BODY EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        model.to('cpu')



# APFL
def apfl_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # get current personalized model based on the previous global omdel
    personalized_model = copy.deepcopy(model)
    personalized_model.apply(partial(set_lambda, lam=args.apfl_constant))
    
    # prepare model
    personalized_model.train()
    personalized_model.to(args.device)
    
    # update local model only
    for name, parameter in personalized_model.named_parameters():
        if 'local' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False

    # prepare optimizer       
    local_optimizer = optimizer(
        [parameter for name, parameter in personalized_model.named_parameters() if 'local' in name], 
        lr=lr, 
        momentum=0.9
    )
    
    # wait, get the global model, too
    model.apply(partial(set_lambda, lam=0.0))

    # prepare model
    model.train()
    initiate_model(model, args)
    
    # update global model only
    for name, parameter in model.named_parameters():
        if 'local' in name:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True
    
    # prepare optimizer       
    global_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if 'local' not in name], 
        lr=lr, 
        momentum=0.9
    )
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)
    
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
            for param in model.parameters(): param.grad = None
            for param in personalized_model.parameters(): param.grad = None
            global_loss.backward(); local_loss.backward()
            global_optimizer.step(); local_optimizer.step()
            
            # get loss
            losses += global_loss.item() + local_loss.item()
            
            # temporarliy merge into personalized model for metrics
            with torch.no_grad():
                temp_model = copy.deepcopy(model)
                local_updated = {k: v for k, v in personalized_model.state_dict().items() if 'local' in k}
                global_updated = model.state_dict()
                global_updated.update(local_updated)
                temp_model.load_state_dict(global_updated)
                
                temp_model.apply(partial(set_lambda, lam=args.apfl_constant))
                outputs = temp_model(inputs)
                
            # get accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        # merge updated local model with updated global model 
        local_updated = {k: v for k, v in personalized_model.state_dict().items() if 'local' in k}
        global_updated = model.state_dict()
        global_updated.update(local_updated)
        model.load_state_dict(global_updated)
        model = model.to('cpu')



# Ditto
def ditto_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    assert args.tau > 0, '[ERROR] argument `tau` should be properly assigned!'
    
    # prepare model
    model.train()
    initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)
    
    # get the global model first
    model.apply(partial(set_lambda, lam=0.0))
    
    # update global model only
    for name, parameter in model.named_parameters():
        if 'local' in name:
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
            for param in model.parameters(): param.grad = None
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
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [GLOBAL MODEL EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
        
    # then, get the local model
    model.apply(partial(set_lambda, lam=1.0))
    
    # update local model only
    for name, parameter in model.named_parameters():
        if 'local' in name:
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
                if 'local' not in name:
                    continue
                prox += (weights[name[:-6]] - weights[name]).norm(2)
            local_loss += args.mu * prox
            
            # update
            for param in model.parameters(): param.grad = None
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
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [LOCAL MODEL EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        model.to('cpu')



# pFedMe
def pfedme_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # prepare model
    model.train()
    initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)
    
    # get the local model first
    model.apply(partial(set_lambda, lam=1.0))

    # update global model only
    for name, parameter in model.named_parameters():
        if 'local' in name:
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
                if 'local' not in name:
                    continue
                prox += (weights[name[:-6]] - weights[name]).norm(2)
            loss += args.mu * prox

            # update
            for param in model.parameters(): param.grad = None
            loss.backward()
            optimizer.step() 

            # get the final loss
            losses += loss.item()

            # get the final accuracy
            accs = get_accuracy(outputs, targets, (1, 5))
            acc1 += accs[0].item(); acc5 += accs[-1].item()

            # get calibration errors
            ces = CalibrationError()(outputs, targets)
            ece += ces[0].item(); mce += ces[-1].item()

            # clear cache
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:
            # update global model based on the \delta-approximated local model weights (line 8 of Algorihtm 1 in the paper)
            weights = model.state_dict()
            for name in weights.keys():
                if 'local' not in name:
                    continue
                weights[name[:-6]] = weights[name[:-6]] - lr * args.mu * (weights[name[:-6]] - weights[name])

            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        model = model.to('cpu')

    
    
    
# SuPerFed-MM (Model-wise Mixture) & SuPerFed-LM (Layer-wise Mixture)
def superfed_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs, start_mix):            
    # retrieve mode (MM or LM)
    mode = args.algorithm[-2:]
    
    # set global model for a regularization
    if args.mu > 0:
        previous_global_model = copy.deepcopy(model)
        for parameter in previous_global_model.parameters(): parameter.requires_grad = False
    
    # prepare model
    model.train()
    initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)

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
                for p_local, p_global in zip(model.parameters(), previous_global_model.parameters()): 
                    prox += (p_local - p_global.to(args.device)).norm(2)
                loss += args.mu * prox

            # subspace construction
            if start_mix:
                weights = model.state_dict()
                numerator, norm_1, norm_2 = 0, 0, 0
                for name in weights.keys():
                    if 'local' not in name:
                        continue
                    numerator += (weights[name[:-6]] * weights[name]).sum()
                    norm_1 += weights[name[:-6]].pow(2).sum()
                    norm_2 += weights[name].pow(2).sum()
                cos_sim = numerator.pow(2) / (norm_1 * norm_2)
                loss += args.nu * cos_sim
            
            # update
            for param in model.parameters(): param.grad = None
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
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:            
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        model = model.to('cpu')

    
###################
# Evaluate models #
###################
@torch.no_grad()
def basic_evaluate(identifier, args, model, criterion, dataset):
    # prepare model
    model.eval()
    initiate_model(model, args)
    
    if identifier is not None: # personalized model evaluation
        if args.algorithm in ['apfl', 'ditto', 'pfedme']:
            model.apply(partial(set_lambda, lam=args.apfl_constant if args.algorithm == 'apfl' else 1.0))
        elif args.algorithm in ['superfed-mm', 'superfed-lm']:
            model.apply(partial(set_lambda, lam=0.3))
    else: # global model evaluation
        model.apply(partial(set_lambda, lam=0.0))

    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=False)
    
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
        if 'cuda' in args.device: torch.cuda.empty_cache()
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
        model.to('cpu')
    return losses, acc1, acc5, ece, mce



@torch.no_grad()
def superfed_evaluate(identifier, args, model, criterion, dataset, current_round):
    results = []
    for lam in np.arange(0.0, 1.1, 0.1):
        # prepare model
        model.eval()
        initiate_model(model, args)
        
        # make dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=False)
    
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
            if 'cuda' in args.device: torch.cuda.empty_cache()
        else:
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            results.append([losses, acc1, acc5, ece, mce])
    
    # get best metrics to print
    results = torch.tensor(results).T
    best_acc_idx = results[1].argmax()
    losses, acc1, acc5, ece, mce = results[:, best_acc_idx].squeeze()
    
    print(f'\t[EVALUATION - CLIENT ({str(identifier).zfill(4)})] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    model.to('cpu')
    return results
