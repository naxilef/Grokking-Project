import torch

from data.modular_arith import make_modular_dataset, modular_addition, split_dataset
from models.modular_mlp import ModularMLP

def train(model, x_train, y_labels, loss_fct, optimizer, device, epochs = 1000):
    model.train()
    losses = [] 

    x_train = x_train.to(device)
    y_labels = y_labels.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model(x_train)
        y_onehot = torch.nn.functional.one_hot(y_labels, num_classes=model.output_dim).float()
        loss = loss_fct(outputs, y_onehot)
        

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}')

    return losses    

def evaluate(model, x_test, y_test, device):
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        outputs = model(x_test)
        predictions = torch.argmax(outputs, dim=1)

        correct = (predictions == y_test).sum().item()
        accuracy = correct / y_test.size(0)

        print(f"Accuracy: {accuracy * 100:.2f}%")
    
        return accuracy
    
def run(p, hidden_dim, alpha=0.5, seed=0, activation="quadratic", lr=1e-3, epochs=1000):
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tensors = make_modular_dataset(p, modular_addition)
    train_dataset, test_dataset = split_dataset(tensors.x, tensors.y, alpha=alpha,seed=seed)

    model = ModularMLP(p=p, hidden_dim=hidden_dim, activation=activation).to(device)
    loss_fct = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train(model=model, x_train=train_dataset.x, y_labels=train_dataset.y, loss_fct=loss_fct, optimizer=optimizer, device=device, epochs=epochs)

    results = evaluate(model=model, x_test=test_dataset.x, y_test=test_dataset.y, device=device)

    return model, results


if __name__ == "__main__":
    model, results = run(
        p=97,
        hidden_dim=512,
        alpha=0.45,
        seed=42,
        activation="quadratic",
        lr=1e-2,
        epochs=2000,
        )
    


    