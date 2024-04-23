import torch
import torch.nn as nn
import torch.optim as optim
from dataset_generation import train_loader, validation_loader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Define the model
class ConvNet(nn.Module):
    def __init__(self, conv_layers, dense_layers, num_classes, dropout):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential()

        # Initial number of filters and input channels
        filters = 32
        in_channels = 3  # RGB images have 3 channels

        # Adding convolutional layers dynamically
        for i in range(conv_layers):
            self.layers.add_module(f"conv{i}", nn.Conv2d(in_channels, filters, kernel_size=3, padding=1))
            self.layers.add_module(f"relu{i}", nn.ReLU())
            self.layers.add_module(f"pool{i}", nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = filters
            filters *= 2  # Double the number of filters after each layer

        # Flattening the output of the last pooling layer
        self.layers.add_module("flatten", nn.Flatten())

        # Calculate the total number of features after flattening
        # Assuming the input images are 256x256
        self.final_conv_output_size = (256 // (2 ** conv_layers)) ** 2 * in_channels

        # Dense layers
        previous_output_size = self.final_conv_output_size
        for j in range(dense_layers):
            self.layers.add_module(f"dense{j}", nn.Linear(previous_output_size, 64))
            self.layers.add_module(f"relu_dense{j}", nn.ReLU())
            previous_output_size = 64
            if dropout:
                self.layers.add_module("dropout", nn.Dropout(0.2))

        # Output layer
        self.layers.add_module("output", nn.Linear(previous_output_size, num_classes))

    def forward(self, x):
        return self.layers(x)

def train_model(conv_layers, dense_layers, num_classes, learning_rate, epochs, dropout, preset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet(conv_layers, dense_layers, num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup TensorBoard
    writer = SummaryWriter(f'./runs/{preset_name}')
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc='Training'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        train_loss /= total
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_accuracy)

        # Log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # Validation phase
        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(validation_loader, desc='Validation'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_accuracy = correct / total
        validation_loss /= total
        metrics['val_loss'].append(validation_loss)
        metrics['val_acc'].append(validation_accuracy)

        # Log validation metrics
        writer.add_scalar('Loss/validation', validation_loss, epoch)
        writer.add_scalar('Accuracy/validation', validation_accuracy, epoch)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy * 100:.2f}%, '
              f'Val Loss: {validation_loss:.4f}, Val Acc: {validation_accuracy * 100:.2f}%')

    writer.close()
    torch.save(model.state_dict(), f'{preset_name}_model.pt')
    return metrics

def main():
    results = []
    parameter_presets = {
        'Preset1': (2, 1, 200, 0.0001, 10, True),
        'Preset2': (3, 1, 200, 0.0001, 10, True),
        'Preset3': (4, 1, 200, 0.0001, 10, True),
        'Preset4': (5, 1, 200, 0.0001, 10, True),
        'Preset5': (6, 1, 200, 0.0001, 10, True),
    }

    for preset_name, parameters in parameter_presets.items():
        metrics = train_model(*parameters, preset_name)
        validation_accuracy = metrics['val_acc']
        results.append((validation_accuracy[-1], preset_name))
        plt.plot(validation_accuracy, label=preset_name)

    results.sort(reverse=True, key=lambda x: x[0])
    for result in results:
        print(f'Preset: {result[1]}, Final accuracy: {result[0]:.2f}%')

    plt.title('Model accuracy for birds')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('birds_accuracy.png')
    plt.show()

if __name__ == '__main__':
    main()