import argparse
import importlib
from utils import test
from utils.train import Trainer
from utils.test import Tester
import torch


def load_model(model_name):
    """Dynamically load the model class from models directory."""
    try:
        module = importlib.import_module(f'models.{model_name}')
        # assuming the class is named 'Model'
        model_class = getattr(module, 'Model')
        return model_class
    except ModuleNotFoundError:
        print(f"Model {model_name} not found in models/ directory.")
        exit(1)
    except AttributeError:
        print(f"'Model' class not found in {model_name}.py")
        exit(1)


def get_dataloaders(name):
    """Imports the correct dataloader"""
    train_dataloader = None
    test_dataloader = None
    val_dataloader = None
    match name:
        case "proof_of_concept":
            from data.proof_of_concept.dataloaders import train_dataloader, test_dataloader, val_dataloader
    if train_dataloader is None or test_dataloader is None or val_dataloader is None:
        raise ValueError("Provide a valid dataset")
    return train_dataloader, test_dataloader, val_dataloader


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a specified model.")
    parser.add_argument('--train', action='store_true',
                        help="Flag to train the model.")
    parser.add_argument('--model', type=str, required=True,
                        help="Specify the model name (e.g., 'model_a').")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs to train (default: 10).")
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop', 'adamw'],
                        help="Optimizer to use (default: 'adam').")
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help="Learning rate for the optimizer (default: 0.001).")
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['step', 'exponential', 'cosine'],
                        help="Learning rate scheduler to use (default: None).")
    parser.add_argument('--step-size', type=int, default=10,
                        help="Step size for StepLR (default: 10, required if using 'step' scheduler).")
    parser.add_argument('--gamma', type=float, default=0.1,
                        help="Gamma for StepLR (default: 0.1).")
    parser.add_argument('--dataset', type=str, default="proof_of_concept",
                        help="Which dataset to load. Possible arguments: 'proof_of_concept'.")

    args = parser.parse_args()

    if args.train:
        # Load the specified model class
        model_class = load_model(args.model)

        # Initialize model instance
        model = model_class().to('cuda' if torch.cuda.is_available() else 'cpu')

        # Get the data loaders (implement this function based on your dataset)
        train_loader, test_loader, val_loader = get_dataloaders(args.dataset)

        # Initialize the Trainer
        trainer = Trainer(model, train_loader, test_loader,
                          val_loader, args.epochs)

        # Set up optimizer
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate)

        # default:

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        trainer.add_optimizer(optimizer)

        # Set up criterion (assuming CrossEntropyLoss is used; adjust as needed)
        criterion = torch.nn.CrossEntropyLoss()
        trainer.add_criterion(criterion)

        # Set up scheduler if specified
        if args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.step_size, gamma=args.gamma)
            trainer.add_scheduler(scheduler)
        elif args.scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.gamma)
            trainer.add_scheduler(scheduler)
        elif args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs)
            trainer.add_scheduler(scheduler)

        # Start training
        trainer.train()  # Assuming you have a train method in the Trainer class


if __name__ == '__main__':
    main()
