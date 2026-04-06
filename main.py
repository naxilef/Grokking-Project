import torch

from data.modular_arith import make_modular_dataset, modular_addition, split_dataset
from models.modular_mlp import ModularMLP

import matplotlib.pyplot as plt

def train(model, x_train, y_train, x_test, y_test,loss_fct, optimizer, device, epochs = 1000):

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    x_test = x_test.to(device)
    y_test = y_test.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(x_train)
        y_train_onehot = torch.nn.functional.one_hot(y_train, num_classes=model.output_dim).float()
        train_loss = loss_fct(outputs, y_train_onehot)
        
        train_loss.backward()

        optimizer.step()

        train_loss_value = train_loss.item()
        train_accuracy = evaluate(
            model, x_train, y_train, device
        )
        test_outputs = model(x_test)
        y_test_onehot = torch.nn.functional.one_hot(y_test, num_classes=model.output_dim).float()
        test_loss_value = loss_fct(test_outputs, y_test_onehot).item()
        test_accuracy = evaluate(
            model, x_test, y_test, device
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss_value)
        history["test_loss"].append(test_loss_value)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss.item():.8f}')

    return history    

def evaluate(model, x, y, device):
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        outputs = model(x)
        
        predictions = torch.argmax(outputs, dim=1)

        correct = (predictions == y).sum().item()
        accuracy = correct / y.size(0)
    
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
    #loss_fct = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = train(model=model, x_train=train_dataset.x, y_train=train_dataset.y, x_test=test_dataset.x, y_test=test_dataset.y,loss_fct=loss_fct, optimizer=optimizer, device=device, epochs=epochs)

    results = evaluate(model=model, x=test_dataset.x, y=test_dataset.y, device=device)

    return model, history

def plot_losses(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["test_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Figure 1(a): Train and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_train_loss.png")
    plt.show()

def plot_accuracy(history):

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
    plt.plot(history["epoch"], history["test_accuracy"], label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Figure 1(c): Train and Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_train_accuracy.png")
    plt.show()


if __name__ == "__main__":
    model, history = run(
        p=97,
        hidden_dim=512,
        alpha=0.49,
        seed=21,
        activation="quadratic",
        lr=1e-2,
        epochs=2000,
        )
    plot_losses(history)
    plot_accuracy(history)


    