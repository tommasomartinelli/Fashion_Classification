import torch
import numpy as np
import matplotlib.pyplot as plt

def train_and_validate(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50, patience=10, plot=True, seed=None):
    # Set seed for reproducibility
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    
    best_valid_loss = np.inf
    best_model = None
    no_improvement = 0
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Iterate over epochs
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        train_loss = running_train_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        running_valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_valid_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_valid += (predicted == labels).sum().item()
                total_valid += labels.size(0)
        
        valid_loss = running_valid_loss / len(valid_loader.dataset)
        valid_accuracy = correct_valid / total_valid
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        
        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy*100:.2f}%")
        
        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"No improvement in validation loss for {patience} epochs. Early stopping...")
                break
    
    # Plotting
    if plot:
        plt.figure(figsize=(10, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
        plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train')
        plt.plot(range(1, len(valid_accuracies)+1), valid_accuracies, label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show() 

    
