from load import load
from preprocess import encoder, scale
from train import train, cross_validation, save_model
import warnings

warnings.filterwarnings("ignore")


def main():
    df = load("../data/train.csv")
    df = encoder(df)
    df = scale(df)
    model = train(df)
    cross_validation(model, df)
    save_model(model, "../models/xgb.pkl")


if __name__ == "__main__":
    main()