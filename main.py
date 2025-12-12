import argparse

from main_scripts.main_training import main_training
from main_scripts.main_evaluation import main_evaluation
from main_scripts.main_cross_evaluation import main_cross_evaluation

def parse_args():
    parser = argparse.ArgumentParser(description="Other script arguments")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--relative_path_to_config')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--artifact_name')
    parser.add_argument('--cross_eval', action='store_true', default=False)
    parser.add_argument('--trained_artifact_name')
    parser.add_argument('--relative_path_to_data_cross_eval_config')
    return parser.parse_args()


def main(args):
    if args.cross_eval:
        if args.train or args.eval:
            print("Cannot specify --cross_eval  with --train or --eval flags.")
        else:
            trained_artifact_name = args.trained_artifact_name
            data_cross_eval_config_path = f"configurations/{args.relative_path_to_data_cross_eval_config}"
            main_cross_evaluation(trained_artifact_name, data_cross_eval_config_path)
    elif args.train and args.eval:
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
        print("Please specify one of the flags: --train  --eval  cross_eval")

if __name__ == "__main__":
    args = parse_args()
    main(args)