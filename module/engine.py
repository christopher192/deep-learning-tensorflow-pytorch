import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def model_training(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
    loss_function: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_accuracy = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        loss = loss_function(y_pred, y)
        train_loss += loss.item() 

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
        train_accuracy += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)

    return train_loss, train_accuracy

def model_testing(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
    loss_function: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval() 
    test_loss, test_accuracy = 0, 0

    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            test_pred_logit = model(x)

            loss = loss_function(test_pred_logit, y)
            test_loss += loss.item()

            test_pred_label = test_pred_logit.argmax(dim = 1)
            test_acc += ((test_pred_label == y).sum().item() / len(test_pred_label))

    test_loss = test_loss / len(dataloader)
    test_accuracy = test_acc / len(dataloader)

    return test_loss, test_accuracy

def start_training(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module, epoch_number: int, device: torch.device) -> Dict[str, List]:
    result = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }
    
    model.to(device)

    for epoch in tqdm(range(epoch_number)):
        train_loss, train_accuracy = model_training(model = model, dataloader = train_dataloader,
            loss_function = loss_function, optimizer = optimizer, device = device)
        test_loss, test_accuracy = model_testing(model = model, dataloader = test_dataloader,
          loss_function = loss_function, device = device)

        print(
          f"Epoch: {epoch + 1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_accuracy:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_accuracy:.4f}"
        )

        result["train_loss"].append(train_loss)
        result["train_accuracy"].append(train_accuracy)
        result["test_loss"].append(test_loss)
        result["test_accuracy"].append(test_accuracy)

    return result