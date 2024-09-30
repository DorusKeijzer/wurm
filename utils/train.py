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
from .utils import plot_loss_and_accuracy, format_time
from time import perf_counter
from contextlib import contextmanager


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, val_dataloader, num_epochs):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        print(f"Training on {self.device}")

        self.best_accuracy = 0
        self.best_model = None
        self.accuracies = []
        self.losses = []

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_criterion(self, criterion):
        self.criterion = criterion

    def add_scheduler(self, scheduler):
        self.scheduler = scheduler

    def addmodel(self, model):
        self.model = model

    def get_model_summary_as_string(self, input_size):
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

    def post_training(self):
        """Gets the test accuracy and writes training history to a file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot = plot_loss_and_accuracy(self.losses, self.accuracies)
        resultsdir = os.path.join(os.getcwd(), "results")
        newdir = os.path.join(resultsdir, f"{timestamp}")
        os.mkdir(newdir)

        fig_filename = os.path.join(newdir, "loss.png")
        plot.savefig(fig_filename)

        self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_dataloader:

                images, labels = images.to(self.device), labels.to(
                    self.device)  # Move data to the appropriate device
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy of the self.model on the test images: {
              100 * correct / total:.2f}")

        sum = self.get_model_summary_as_string((3, 224, 224))

        res_filename = os.path.join(newdir, "results.txt")
        with open(res_filename, "w") as t:
            t.write(sum)

            t.write("---------------------------------------------\n")
            t.write(f"Optimizer: {self.optimizer.__class__.__name__}\n")
            t.write(f"Scheduler: {self.scheduler.__class__.__name__}\n")
            t.write(f"Loss: {self.criterion.__class__.__name__}\n")
            t.write("---------------------------------------------\n\n")
            t.write(f"\n\nTest accuracy: {100 * correct / total:.2f}%\n")

            t.write(f"Loss over time: {self.losses}\n")
            t.write(f"Accuracies over time: {self.accuracies}\n\n")
            t.write(f"Best accuracy {self.best_accuracy}")

        model_filename = os.path.join(newdir, "last_model.pth")
        best_filename = os.path.join(newdir, "best_self.model.pth")
        torch.save(self.model.state_dict(), model_filename)
        torch.save(self.model.state_dict(), best_filename)

    def train(self):
        interrupted = False

        def signal_handler(sig, frame):
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGINT, signal_handler)

        for epoch in range(self.num_epochs):
            start = perf_counter()

            self.train_epoch(epoch, interrupted)
            val_loss = self.validate()

            self.update_metrics(epoch, val_loss, start)

        self.post_training()

    def train_epoch(self, epoch, interrupted):
        with self.train_mode():
            running_loss = 0.0
            for i, (images, labels) in enumerate(self.train_dataloader):
                self.print_progress(epoch, i)
                if interrupted:
                    self.handle_interruption()
                    interrupted = False

                running_loss += self.train_batch(images, labels)

        return running_loss

    @contextmanager
    def train_mode(self):
        self.model.train()
        yield
        self.model.eval()

    def train_batch(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self):
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.val_dataloader)
        accuracy = 100 * correct / total
        return val_loss, accuracy

    def update_metrics(self, epoch, val_loss, start_time):
        self.scheduler.step(val_loss)
        lr = self.scheduler.get_last_lr()[0]
        loss = sum(self.losses) / len(self.train_dataloader)
        accuracy = self.accuracies[-1]

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = self.model.state_dict()

        end = perf_counter()
        expected_time_left = (self.num_epochs - epoch - 1) * (end - start_time)

        self.print_epoch_summary(
            epoch, loss, accuracy, val_loss, lr, expected_time_left)

    def print_epoch_summary(self, epoch, loss, accuracy, val_loss, lr, expected_time_left):
        print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
        print(f"Loss: {loss:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.2f}")
        print(f"Learning Rate: {lr:.6f}")
        print(f"Expected time left: {format_time(expected_time_left)}")

    def handle_interruption(self):
        print("\nTraining interrupted. Choose an option:")
        print("1. Save the model")
        print("2. Change learning rate")
        print("3. Exit training (save)")
        print("4. Exit training (don't save)")

        choice = input("Enter your choice: ")

        if choice == '1':
            self.save_model()
        elif choice == '2':
            self.change_learning_rate()
        elif choice == '3':
            self.save_model()
            sys.exit(0)
        elif choice == '4':
            sys.exit(0)
        else:
            print("Invalid choice. Continuing training.")

    def save_model(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def change_learning_rate(self):
        try:
            new_lr = float(input("Enter new learning rate: "))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate changed to {new_lr}")
        except ValueError:
            print("Invalid learning rate. Continuing with current rate.")

    def print_progress(self, epoch, batch):
        print(f"Epoch [{epoch+1}/{self.num_epochs}], "
              f"Batch [{batch*self.train_dataloader.batch_size}/"
              f"{len(self.train_dataloader.dataset)}]", end="\r")
