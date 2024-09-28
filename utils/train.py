from torchsummary import summary
import signal
import torch.nn as nn
import torch.optim as optim
import torch
from datetime import datetime
import os
import io
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import plot_loss_and_accuracy, format_time

from time import perf_counter


class trainer:
    best_accuracy = 0
    best_model = None
    accuracies = []
    losses = []

    def __init__(self, model, train_dataloader, test_dataloader, val_dataloader, num_epochs):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        print(f"Training on {self.device}")

    def add_criterion(self, criterion):
        self.criterion = criterion

    def add_scheduler(self, scheduler):
        self.scheduler = scheduler

    def addmodel(self, model):
        self.model = model

    def post_training(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot = plot_loss_and_accuracy(losses, accuracies)
        resultsdir = os.path.join(os.getcwd(), "results")
        newdir = os.path.join(resultsdir, f"{timestamp}")
        os.mkdir(newdir)

        fig_filename = os.path.join(newdir, "loss.png")
        plot.savefig(fig_filename)

        self.self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_dataloader:

                images, labels = images.to(self.device), labels.to(
                    self.device)  # Move data to the appropriate device
                outputs = self.self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy of the self.model on the test images: {
              100 * correct / total:.2f}")

        def get_self.model_summary_as_string(model, input_size):
            # Redirect stdout to capture the summary
            buffer = io.StringIO()
            sys.stdout = buffer

            summary(self.model, input_size=input_size)

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Get the summary string from buffer
            summary_str = buffer.getvalue()
            buffer.close()

            return summary_str

        sum = get_self.model_summary_as_string(model, (3, 224, 224))

        res_filename = os.path.join(newdir, "results.txt")
        with open(res_filename, "w") as t:
            t.write(sum)

            t.write("---------------------------------------------\n")
            t.write(f"Optimizer: {optimizer.__class__.__name__}\n")
            t.write(f"Scheduler: {scheduler.__class__.__name__}\n")
            t.write(f"Loss: {criterion.__class__.__name__}\n")
            t.write("---------------------------------------------\n\n")
            t.write(f"\n\nTest accuracy: {100 * correct / total:.2f}%\n")

            t.write(f"Loss over time: {losses}\n")
            t.write(f"Accuracies over time: {accuracies}\n\n")
            t.write(f"Best accuracy {best_accuracy}")

        self.model_filename = os.path.join(newdir, "last_model.pth")
        best_filename = os.path.join(newdir, "best_self.model.pth")
        torch.save(self.model.state_dict(), model_filename)
        torch.save(self.model.state_dict(), best_filename)

    def train(self):
        # Variable to track the interruption
        interrupted = False

        def signal_handler(sig, frame):
            global interrupted
            interrupted = True
        # Register the signal handler
        signal.signal(signal.SIGINT, signal_handler)

        def handle_interruption():
            print("\nTraining interrupted. Choose an option:")
            print("1. Save the self.model")
            print("2. Save progress graph")
            print("3. Change learning rate")
            print("4. Exit training (save)")
            print("5. Exit training (don't save)")

            choice = input("Enter your choice: ")

            if choice == '1':
                torch.save(self.model.state_dict(), 'model.pth')
                print("self.model saved.")
            elif choice == '2':
                evaluate_self.model()
            elif choice == '3':
                new_lr = float(input("Enter new learning rate: "))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Learning rate changed to {new_lr}")
            elif choice == '4':
                print("Exiting training...")
                post_training()
                sys.exit(0)
            elif choice == '5':
                print("Exiting training...")
                sys.exit(0)
            else:
                print("Invalid choice. Continuing training.")

               self.model = Flower_model(525).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5)

        losses = []
        accuracies = []
        old_lr = scheduler.get_last_lr()
        # Training loop
        for epoch in range(num_epochs):
            start = perf_counter()
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_dataloader):
                print(f"Epoch [ {epoch+1}/{num_epochs} ], Batch [ {
                      i*train_dataloader.batch_size}/{len(train_dataloader.dataset)} ]", end="\r")
                if interrupted:
                    handle_interruption()
                    interrupted = False  # Reset the flag after handling
                images, labels = images.to(self.device), labels.to(
                    self.device)  # Move data to the appropriate device
                optimizer.zero_grad()  # Zero the parameter gradients
                outputs = self.model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights

                running_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.val_dataloader:
                    images, labels = images.to(self.device), labels.to(
                        self.device)  # Move data to the appropriate device
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(self.val_dataloader)
            scheduler.step(val_loss)

            lr = scheduler.get_last_lr()
            loss = running_loss / len(self.train_dataloader)
            accuracy = 100 * correct / total
            if accuracy > self.best_accuracy:
                best_accuracy = accuracy
                self.best_model = self.model
            losses.append(loss)
            accuracies.append(accuracy)

            end = perf_counter()

            expected_time_left = (self.num_epochs - epoch - 1) * (end - start)

            print(f"Epoch [ {epoch+1}/{self.num_epochs} ]")
            print(f"Loss:                {loss:>7.2f}")
            print(f"Accuracy:            {accuracy:>7.2f} %")
            print(f"Validation Loss:     {val_loss:>7.2f}")
            if format_time(expected_time_left):
                print(f"Expected time left:     {
                      format_time(expected_time_left):>7}")
            if old_lr != lr:
                old_lr = lr
                print(f"Plateaued, decreased learning rate to {old_lr}")
            print("")
        post_training()
