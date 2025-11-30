import argparse

from main_scripts.main_training import main_training
from main_scripts.main_evaluation import main_evaluation

def parse_args():
    parser = argparse.ArgumentParser(description="Other script arguments")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--relative_path_to_config')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--artifact_name')
    return parser.parse_args()


def main(args):
    if args.train and args.eval:
        config_path = f"configurations/{args.relative_path_to_config}"
        artifact_name = main_training(config_path)
        artifact_name += ":latest"
        main_evaluation(artifact_name)

    elif args.train:
        config_path = f"configurations/{args.relative_path_to_config}"
        artifact_name = main_training(config_path)
    
    elif args.eval:
        artifact_name = args.artifact_name
        main_evaluation(artifact_name)
    
    else:
        print("Please specify either --train or --eval flag.")

if __name__ == "__main__":
    args = parse_args()
    main(args)