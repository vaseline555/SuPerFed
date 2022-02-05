import gc
import copy
import torch

from .utils import initiate_model, get_accuracy, CalibrationError



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
        model.to('cpu')
    return model



# FedProx
def regularized_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # set local model
    local_model = copy.deepcopy(model)
    local_model.train()
    local_model = initiate_model(local_model, args)
    
    # set global model for a regularization
    previous_optimal_model = copy.deepcopy(model)
    for parameter in previous_optimal_model.parameters(): parameter.requires_grad = False
    previous_optimal_model.eval(); previous_optimal_model.to(args.device)
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
                prox += (p_local - p_global).norm(2)
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
        local_model.to('cpu')
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
        model.to('cpu')
    return model



# APFL
def apfl_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # get current personalized model based on the previous global omdel
    personalized_model = copy.deepcopy(model)
    for module in personalized_model.modules():
        if (
            isinstance(module, torch.nn.Conv2d) 
            or isinstance(module, torch.nn.BatchNorm2d)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.LSTM)
            or isinstance(module, torch.nn.Embedding)
        ):
            setattr(module, 'alpha', args.alpha)
    
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
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Conv2d) 
            or isinstance(module, torch.nn.BatchNorm2d)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.LSTM)
            or isinstance(module, torch.nn.Embedding)
        ):
            setattr(module, 'alpha', 0.0)
    
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
            loss = criterion()(outputs, targets); local_loss = criterion()(local_outputs, targets)

            # update
            global_optimizer.zero_grad(); local_optimizer.zero_grad()
            loss.backward(); local_loss.backward()
            global_optimizer.step(); local_optimizer.step()
            
            # get loss
            losses += loss.item() + local_loss.item()
            
            # temporarliy merge into personalized model for metrics
            with torch.no_grad():
                temp_model = copy.deepcopy(model)
                local_updated = {k: v for k, v in personalized_model.state_dict().items() if '_local' in k}
                global_updated = model.state_dict()
                global_updated.update(local_updated)
                temp_model.load_state_dict(global_updated)
                
                for module in temp_model.modules():
                    if (
                        isinstance(module, torch.nn.Conv2d) 
                        or isinstance(module, torch.nn.BatchNorm2d)
                        or isinstance(module, torch.nn.Linear)
                        or isinstance(module, torch.nn.LSTM)
                        or isinstance(module, torch.nn.Embedding)
                    ):
                        setattr(module, 'alpha', args.alpha)
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [FIRST EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del global_optimizer, local_optimizer; gc.collect()
        
        # merge updated local model with updated global model 
        local_updated = {k: v for k, v in personalized_model.state_dict().items() if '_local' in k}
        global_updated = model.state_dict()
        global_updated.update(local_updated)
        model.load_state_dict(global_updated)
        del personalized_model, local_updated, global_updated; gc.collect()
        
        model.to('cpu')
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
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Conv2d) 
            or isinstance(module, torch.nn.BatchNorm2d)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.LSTM)
            or isinstance(module, torch.nn.Embedding)
        ):
            setattr(module, 'alpha', 0.0)
    
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

    # update global model 
    for e in range(args.E if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)

            # update
            global_optimizer.zero_grad()
            loss.backward()
            global_optimizer.step() 
            
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [FIRST EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del global_optimizer; gc.collect()
        
    # then, get the local model
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Conv2d) 
            or isinstance(module, torch.nn.BatchNorm2d)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.LSTM)
            or isinstance(module, torch.nn.Embedding)
        ):
            setattr(module, 'alpha', 1.0)
    
    # update local model only
    for name, parameter in model.named_parameters():
        if '_local' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer       
    local_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if '_local' in name], 
        lr=lr, 
        momentum=0.9
    )
    
    # then update local model
    for e in range(args.tau if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)

            # update
            local_optimizer.zero_grad()
            loss.backward()
            local_optimizer.step() 
            
            # calculate regularization term toward optimal global model
            prox = 0.
            weights = model.load_state_dict()
            for name in weights.keys():
                if '_local' in name:
                    continue
                prox += (weights[name] - weights[f'{name}_local']).norm(2)
            else:
                del weights; gc.collect()
            loss += args.mu * prox
            
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
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [LAST EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del local_optimizer; gc.collect()
        model.to('cpu')
    return model



# pFedMe
def pfedme_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    assert args.tau > 0, '[ERROR] argument `tau` should be properly assigned!'
    
    # prepare model
    model.train()
    model = initiate_model(model, args)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)
    
    # get the local model first
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Conv2d) 
            or isinstance(module, torch.nn.BatchNorm2d)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.LSTM)
            or isinstance(module, torch.nn.Embedding)
        ):
            setattr(module, 'alpha', 1.0)
    
    # update global model only
    for name, parameter in model.named_parameters():
        if '_local' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # prepare optimizer       
    local_optimizer = optimizer(
        [parameter for name, parameter in model.named_parameters() if '_local' in name], 
        lr=lr, 
        momentum=0.9
    )
    
    # update local model first
    for e in range(args.tau if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # inference
            outputs = model(inputs)
            loss = criterion()(outputs, targets)

            # update
            local_optimizer.zero_grad()
            loss.backward()
            local_optimizer.step() 
            
            # calculate regularization term toward global model
            prox = 0.
            weights = model.load_state_dict()
            for name in weights.keys():
                if '_local' in name:
                    continue
                prox += (weights[name] - weights[f'{name}_local']).norm(2)
            else:
                del weights; gc.collect()
            loss += args.mu * prox
            
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
            weights = model.load_state_dict()
            for name in weights.keys():
                if '_local' in name:
                    continue
                weights[name] -= lr * args.mu * (weights[name] - weights[f'{name}_local'])
            else:
                model.load_state_dict(weights)
                del weights; gc.collect()
            
            # get losses & metrics
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [LAST EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del local_optimizer; gc.collect()
        model.to('cpu')
    return model



# SuPerFed-MM (Model Mixture) & SuPerFed-LM (Layer Mixture)
def superfed_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    dataloader = torch.utils.data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        
    if self.mu > 0:
        # fix global model for calculating a proximity term
        self.global_model = copy.deepcopy(self.model)
        self.global_model.to(self.device)

    for param in self.global_model.parameters():
        param.requires_grad = False

    # update local model
    self.model.train()
    self.model.to(self.device)

    parameters = list(self.model.named_parameters())
    parameters_to_opimizer = [v for n, v in parameters if v.requires_grad]            
    optimizer 

    flag = False
    if epoch is None:
        epoch = self.local_epoch
    else:
        flag = True

    for e in range(epoch):
        for data, labels in self.training_dataloader:
            data, labels = data.float().to(self.device), labels.long().to(self.device)

            # mixing models
            if not start_local_training:
                alpha = 0.0
            else:
                if flag:
                    alpha = 0.5
                else:
                    alpha = np.random.uniform(0.0, 1.0)
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                    setattr(m, f"alpha", alpha)

            # inference
            outputs = self.model(data)
            loss = self.criterion()(outputs, labels)

            # subspace construction
            if start_local_training:
                num, norm1, norm2, cossim = 0., 0., 0., 0.
                for m in self.model.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) and hasattr(m, 'weight1'):
                        num += (m.weight * m.weight1).sum()
                        norm1 += m.weight.pow(2).sum()
                        norm2 += m.weight1.pow(2).sum()
                cossim = self.beta * (num.pow(2) / (norm1 * norm2))
                loss += cossim

            # proximity regularization
            prox = 0.
            for (n, w), w_g in zip(self.model.named_parameters(), self.global_model.parameters()):
                if "weight1" not in n:
                    prox += (w - w_g).norm(2)
            loss += self.mu * (1 - alpha) * prox

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            if self.device == "cuda": torch.cuda.empty_cache() 
    self.model.to("cpu")
    self.model.eval()

    
###################
# Evaluate models #
###################
@torch.no_grad()
def basic_evaluate(identifier, args, model, criterion, dataset):
    # prepare model
    model.eval()
    model.to(args.device)
    
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
def mixed_evaluate(args, model, criterion, dataset):
    pass

@torch.no_grad()
def superfed_evaluate(args, model, criterion, dataset):
    model.eval()
    model.to(device)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    best_acc, best_loss, best_topk = 0, 1000, 0
    for alpha in [0.0]:
        # set alpha for inference
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                setattr(m, f"alpha", alpha)

        accs = []
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += torch.nn.CrossEntropyLoss()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
                accs.append(accuracy(outputs, labels, (5,)))
        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        print(f"[INFO] test_loss: {test_loss:.4f}, test_acc: {test_accuracy:.4f}")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
        if test_loss < best_loss:
            best_loss = test_loss
        if torch.stack(accs).mean(0) > best_topk:
            best_topk = torch.stack(accs).mean(0).item()
    else:   
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                setattr(m, f"alpha", 0.0)

        self.model.to("cpu")
    return best_loss, best_acc, best_topk