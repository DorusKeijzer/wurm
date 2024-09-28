import argparse
import importlib
from utils.train import train
from utils.test import test


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


def main():
    parser = argparse.ArgumentParser(description="Train a specified model.")
    parser.add_argument('--train', action='store_true',
                        help="Flag to train the model.")
    parser.add_argument('--model', type=str, required=True,
                        help="Specify the model name (e.g., 'model_a').")
    parser.add_argument('--test', action='store_true',
                        help="Flag to test the model.")
    args = parser.parse_args()

    if args.test:
        # Load the specified model class dynamically
        model_class = load_model(args.model)

        # Initialize model instance
        model = model_class()

        # Train the model (you can pass more arguments as needed)
        test(model)

    if args.train:
        # Load the specified model class dynamically
        model_class = load_model(args.model)

        # Initialize model instance
        model = model_class()

        # Train the model (you can pass more arguments as needed)
        train(model)


if __name__ == '__main__':
    main()
