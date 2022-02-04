import gc
import copy
import torch

from .utils import get_accuracy, CalibrationError



###############################
# Update models in the client #
###############################
def basic_update(identifier, args, model, criterion, dataset, optimizer, lr, epochs):
    # prepare model
    model.train()
    model.to(args.device)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)

    # prepare optimizer       
    optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9)
    print(len(dataloader))
    # main loop
    for e in range(args.E if epochs is None else epochs):
        # track loss and metrics
        losses, acc1, acc5, ece, mce = 0, 0, 0, 0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            print('하고이써!')
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
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        model.to('cpu')
    return model

def fedrep_update(args, model, criterion, dataset, optimizer, lr, epochs):                        
    # prepare model
    model.train()
    model.to(args.device)
    
    # make dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.B, shuffle=True, pin_memory=True)

    # prepare optimizer       
    optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9)
    
    # update head (penultimate layer) first
    for name, parameter in model.named_parameters():
        if 'classifier' in name:
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    
    # head fine-tuning loop
    for e in range(args.tau):
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
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [HEAD EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    
    # then, update the body
    for name, parameter in model.named_parameters():
        if 'classifier' in name:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True
    
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
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [BODY EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        model.to('cpu')
    return model

def regularized_update(args, model, criterion, dataset, optimizer, lr, epochs):
    # set local model
    local_model = copy.deepcopy(model)
    local_model.train(); local_model.to(args.device)
    
    # set global model for a regularization
    global_model = copy.deepcopy(model)
    for parameter in global_model.parameters(): parameter.requires_grad = False
    global_model.eval(); global_model.to(args.device)
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
            for p_local, p_global in zip(local_model.parameters(), global_model.parameters()): 
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
            losses /= len(dataloader)
            acc1 /= len(dataloader)
            acc5 /= len(dataloader)
            ece /= len(dataloader)
            mce /= len(dataloader)
            print(f'\t[TRAINING - CLIENT ({str(identifier).zfill(4)})] [EPOCH: {str(e).zfill(2)}] Loss: {losses:.4f}, Top1 Acc.: {acc1:.4f}, Top5 Acc.: {acc5:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f}')
    else:
        del global_model; gc.collect()
        local_model.to('cpu')
    return local_model

def superfed_update(args, model, criterion, dataset, optimizer, lr, epoch, mode='lm'):
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
def global_evaluate(args, model, criterion, dataset):
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