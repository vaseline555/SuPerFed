import copy
import torch
import numpy as np

from functools import partial

from .utils import initiate_model, get_accuracy, CalibrationError, set_lambda



###############################
# Update models in the client #
###############################
# FedAvg, LG-FedAvg, FedPer
def basic_update(identifier, args, model, criterion, dataset, optimizer, lr):
    # prepare model
    model.train()
    initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)

    # prepare optimizer       
    optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # main loop
    for e in range(args.E):
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    model.to('cpu')

    
    
# FedProx
def fedprox_update(identifier, args, model, criterion, dataset, optimizer, lr):
    # set local model
    model.train()
    initiate_model(model, args)
    
    # set global model for a regularization
    previous_global_model = copy.deepcopy(model)
    for parameter in previous_global_model.parameters(): parameter.requires_grad = False

    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)

    # prepare optimizer       
    optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    # main loop
    for e in range(args.E):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)
            
            # calculate proximity regularization
            prox = 0.
            for name, param in model.named_parameters():
                if ('bias' in name)  or ('weight' not in name): continue
                prox += (param - previous_global_model.get_parameter(name)).norm(2)
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    model.to('cpu')

    

# SCAFFOLD
# use the implementation of a local model as control variates instead... (which is initiailzed to zeros)
def scaffold_update(identifier, args, model, criterion, dataset, optimizer, lr):
    # prepare model
    model.train()
    initiate_model(model, args)
    model.apply(partial(set_lambda, lam=0.0))
    
    # set global model for a regularization
    previous_global_model = copy.deepcopy(model)
    for parameter in previous_global_model.parameters(): parameter.requires_grad = False
    
    # only update a global model
    for name, parameter in model.named_parameters():
        if 'local' not in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)

    # prepare optimizer       
    optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if 'local' not in name], 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )

    # main loop
    for e in range(args.E):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)
            
            # update parameters
            for param in model.parameters(): param.grad = None
            loss.backward()
            optimizer.step() 
            
            # update parameters by reflecting control varaites
            for name, param in model.named_parameters():
                if ('local' in name) or ('bias' in name) or ('weight' not in name): continue
                param = param - lr * (previous_global_model.get_parameter(f'{name}_local') - model.get_parameter(f'{name}_local'))
            
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
            # update control varaites (c^+_i - c_i)
            for name, param in model.named_parameters():
                if ('local' in name) or ('running' in name) or ('batch' in name): continue
                param = - previous_global_model.get_parameter(name) + (previous_global_model.get_parameter(name.replace('_local', '')) - model.get_parameter(name.replace('_local', ''))) / (args.E * lr)
                
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    model.to('cpu')
    
    
    
# FedRep
def fedrep_update(identifier, args, model, criterion, dataset, optimizer, lr):
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
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    # head fine-tuning loop
    for e in range(args.E):
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [HEAD EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del head_optimizer
        
    # then, update the body
    for name, parameter in model.named_parameters():
        if 'classifier' not in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer 
    body_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if 'classifier' not in name], 
        lr=lr, 
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # body adaptation loop
    for e in range(args.tau):
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [BODY EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    model.to('cpu')



# APFL
def apfl_update(identifier, args, model, criterion, dataset, optimizer, lr):
    # set w
    model.apply(partial(set_lambda, lam=0.0))

    ## prepare model
    model.train()
    initiate_model(model, args)
    
    # update global model only
    for name, parameter in model.named_parameters():
        if 'local' not in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer       
    global_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if 'local' not in name], 
        lr=lr, 
        momentum=0.9
    )
    
    # set \bar{v}
    ## get current personalized model based on the previous global model
    personalized_model = copy.deepcopy(model)
    personalized_model.apply(partial(set_lambda, lam=args.apfl_constant))

    ## update local model only
    for name, parameter in personalized_model.named_parameters():
        if 'local' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False

    ## prepare optimizer       
    local_optimizer = optimizer(
        [parameter for name, parameter in personalized_model.named_parameters() if 'local' in name], 
        lr=lr, 
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)
    
    # update global model & personalized model
    for e in range(args.E):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            # global model (\w)
            # inference
            outputs = model(inputs); 
            global_loss = criterion()(outputs, targets)

            # update
            for param in model.parameters(): param.grad = None
            global_loss.backward()
            global_optimizer.step()
            
            # get loss
            losses += global_loss.item()
            
            # local model (\v)
            # inference
            local_outputs = personalized_model(inputs)
            local_loss = criterion()(local_outputs, targets)
            
            # update
            for param in personalized_model.parameters(): param.grad = None
            local_loss.backward()
            local_optimizer.step()
            
            # get loss
            losses += local_loss.item()
            
            # evaluate with personalized model (\bar{v}) temporarliy merge into personalized model (\bar(v)) for metrics
            with torch.no_grad():
                temp_model = copy.deepcopy(model)
                local_updated = {k: v for k, v in personalized_model.state_dict().items() if 'local' in k}
                global_updated = temp_model.state_dict()
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        # merge updated local model with updated global model 
        local_updated = {k: v for k, v in personalized_model.state_dict().items() if 'local' in k}
        global_updated = model.state_dict()
        global_updated.update(local_updated)
        model.load_state_dict(global_updated)
    model.to('cpu')



# Ditto
def ditto_update(identifier, args, model, criterion, dataset, optimizer, lr):
    assert args.tau > 0, '[ERROR] argument `tau` should be properly assigned!'
    # prepare model
    model.train()
    initiate_model(model, args)
    
    # global model (w_0)
    previous_global_model = copy.deepcopy(model)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)
    
    # get the global model first
    model.apply(partial(set_lambda, lam=0.0))
    
    # update global model only
    for name, parameter in model.named_parameters():
        if 'local' not in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer       
    global_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if 'local' not in name], 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )

    # update global model 
    for e in range(args.E): # updates global model for E local iterations
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [GLOBAL MODEL EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
        
    # then, get the local model
    model.apply(partial(set_lambda, lam=1.0))
    
    # update local model only
    for name, parameter in model.named_parameters():
        if 'local' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer       
    local_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if 'local' in name], 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    # then update local model
    for e in range(args.tau): # update local model for \tau local iterations
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            local_loss = criterion()(outputs, targets)
            
            # get loss
            losses += local_loss.item()
            
            # calculate regularization term toward optimal global model
            prox = 0.
            for name, param in model.named_parameters():
                if ('local' not in name) or ('weight' not in name): continue
                prox += (param - previous_global_model.get_parameter(name.replace('_local', ''))).norm(2)
            local_loss += args.mu * prox            
            
            # update
            for param in model.parameters(): param.grad = None
            local_loss.backward()
            local_optimizer.step() 
            
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [LOCAL MODEL EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    model.to('cpu')



# pFedMe
# https://github.com/CharlieDinh/pFedMe/blob/caf24b1f954d4381bd9b4d104ee7eff389e1489b/FLAlgorithms/users/userpFedMe.py
def pfedme_update(identifier, args, model, criterion, dataset, optimizer, lr):
    # prepare model
    model.train()
    initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)
    
    # get the local model first
    model.apply(partial(set_lambda, lam=1.0))

    # update local model
    for name, parameter in model.named_parameters():
        if 'local' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer       
    optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if 'local' in name], 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
            
    # update local model
    for e in range(args.E):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)

            # get the loss
            losses += loss.item()

            # calculate regularization term toward global model
            prox = 0.
            for name, param in model.named_parameters():
                if ('local' not in name) or ('weight' not in name): continue
                prox += (param - model.get_parameter(name.replace('_local', '')).detach()).norm(2)
            loss += args.mu * prox        
            
            # update
            for param in model.parameters(): param.grad = None
            loss.backward()
            optimizer.step()

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
            current_state = copy.deepcopy(model.state_dict())
            for name in current_state.keys():
                if ('local' in name) or ('running' in name) or ('batch' in name): continue
                current_state[name] = current_state[name] - lr * args.mu * (current_state[name] - current_state[f'{name}_local'])
            model.load_state_dict(current_state)

            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    model.to('cpu')

    
    
# SuPerFed-MM (Model-wise Mixture) & SuPerFed-LM (Layer-wise Mixture)
def superfed_update(identifier, args, model, criterion, dataset, optimizer, lr, start_mix):            
    # retrieve mode (MM or LM)
    mode = args.algorithm[-2:]
    
    # prepare model
    model.train()
    initiate_model(model, args)
    
    # set global model for a regularization
    if args.mu > 0:
        previous_global_model = copy.deepcopy(model)
        for param in previous_global_model.parameters(): param.requires_grad = False
        
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True)
    
    # update federated model before start_mix is set
    if start_mix:
        for name, param in model.named_parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if 'local' not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            model.apply(partial(set_lambda, lam=0.0))
            
    # prepare optimizer       
    optimizer = optimizer(
        model.parameters() if start_mix else [param for name, param in model.named_parameters() if 'local' not in name], 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    # update local model first
    for e in range(args.E):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            if start_mix: # instant model mixing according to mode
                if mode == 'mm':
                    model.apply(partial(set_lambda, lam=np.random.uniform(0.0, 1.0)))
                elif mode == 'lm':
                    model.apply(partial(set_lambda, lam=None, layerwise=True))
                
            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)
            
            # calculate proximity regularization term toward global model
            if args.mu > 0:
                prox = 0.
                for name, param in model.named_parameters(): 
                    if 'local' not in name: continue
                    prox += (model.get_parameter(name.replace('_local', '')) - previous_global_model.get_parameter(name.replace('_local', ''))).norm(2)
                loss += args.mu * prox

            # subspace construction
            if start_mix:
                numerator, norm_1, norm_2 = 0, 0, 0
                for name, param_l in model.named_parameters():
                    if 'local' not in name: continue
                    param_g = model.get_parameter(name.replace('_local', ''))
                    numerator += (param_g * param_l).add(1e-6).sum()
                    norm_1 += param_g.pow(2).sum()
                    norm_2 += param_l.pow(2).sum()
                cos_sim = numerator.pow(2).div(norm_1 * norm_2)
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e + 1).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    model.to('cpu')

    
###################
# Evaluate models #
###################
@torch.no_grad()
def basic_evaluate(identifier, args, model, criterion, dataset):
    # prepare model
    model.eval()
    initiate_model(model, args)
    
    # get global or personalized model
    if args.algorithm in ['apfl', 'pfedme', 'ditto', 'superfed-mm', 'superfed-lm']:
        if identifier is not None: # get local model (i.e., personalized model)
            model.apply(partial(set_lambda, lam=args.apfl_constant if args.algorithm == 'apfl' else 1.0))
        else: # get global model
            model.apply(partial(set_lambda, lam=0.0))
    elif args.algorithm in ['scaffold']:
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
    
    print(f'\t[EVALUATION - CLIENT ({str(identifier).zfill(4)})] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}, LAMBDA: {0 + 0.1 * best_acc_idx:.2f}')
    model.to('cpu')
    return results
