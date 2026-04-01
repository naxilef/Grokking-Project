import torch

from data.modular_arith import make_modular_dataset, modular_addition, split_dataset
from models.modular_mlp import ModularMLP

def train(model, x_train, y_labels, loss_fct, optimizer, epochs = 1000):
    model.train()
    losses = [] 

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model(x_train)
        #y_onehot = torch.nn.functional.one_hot(y_labels, num_classes=model.output_dim).float() #
        loss = loss_fct(outputs, y_labels)
        

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.16f}')

    return losses    

def evaluate(model, x_test, y_test):
    with torch.no_grad():
        outputs = model(x_test)
        predictions = torch.argmax(outputs, dim=1)

        correct = (predictions == y_test).sum().item()
        accuracy = correct / y_test.size(0)

        print(f"Accuracy: {accuracy * 100:.2f}%")
    
        return accuracy
    
def run(p, hidden_dim, alpha=0.5, seed=0, activation="quadratic", lr=1e-3, epochs=1000):
    torch.manual_seed(seed)

    tensors = make_modular_dataset(p, modular_addition)
    train_dataset, test_dataset = split_dataset(tensors.x, tensors.y, alpha=alpha,seed=seed)

    model = ModularMLP(p=p, hidden_dim=hidden_dim, activation=activation)
 
    loss_fct = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    y_train_onehot = torch.nn.functional.one_hot(
        train_dataset.y, num_classes=p
    ).float()

    #loss_fct = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model=model, x_train=train_dataset.x, y_labels=y_train_onehot, loss_fct=loss_fct, optimizer=optimizer, epochs=epochs)

    results = evaluate(model=model, x_test=test_dataset.x, y_test=test_dataset.y)

    return model, results


if __name__ == "__main__":
    model, results = run(
        p=97,
        hidden_dim=512,
        alpha=0.5,
        seed=0,
        activation="quadratic",
        lr=1e-2,
        epochs=200000,
    )
    print("Final test accuracy:", results)
    


    