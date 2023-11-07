import os
from src.load import load
from src.preprocess import preprocess
from src.train import train, cross_validation, save_model
import warnings
import argparse

warnings.filterwarnings("ignore")


def main(model_path="models/xgb.pkl", data_path="data/train.csv", val_split:bool=False, **model_params):
    df = load(data_path)
    X, y = preprocess(df)
    if not model_params:
        model_params = {
            "max_depth":5, "tree_method":"hist", "device":"cuda",
            "objective":"binary:logistic","eval_metric":"auc",
            "random_state":42,
            }
    model = train(X, y, val_split=val_split, **model_params)
    cross_validation(model, X, y)
    save_model(model, model_path)


"""if __name__ == "__main__":
    main()"""


def main_cli(args):
    df = load(args.data_path)
    X, y = preprocess(df)
    model_params = {
        "max_depth":5, "tree_method":"hist", "device":"cuda",
        "objective":"binary:logistic","eval_metric":"auc",
        "random_state":42}
    model = train(X, y, val_split=args.val_split, **model_params)  # Pass val_split parameter
    cross_validation(model, X, y)
    if args.val_split:
        save_model(model, "models/xgb_val_split.pkl")
    else:
        save_model(model, "models/xgb.pkl")
    save_model(model, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")

    parser.add_argument("--data_path", type=str, default="data/train.csv", 
                        help="Path to the input data file (CSV format).")
    parser.add_argument("--model_path", type=str, default="models/xgb.pkl", 
                        help="Path to save the trained model.")
    parser.add_argument("--val_split", action="store_true", 
                        help="Enable validation split during training.")

    args = parser.parse_args()
    main_cli(args)