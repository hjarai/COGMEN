import pickle
import os
import argparse
import torch
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from tqdm import tqdm

import cogmen

log = cogmen.utils.get_logger()


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def main(args):
    data = load_pkl(f"data/{args.dataset}/data_{args.dataset}.pkl")
    print('loaded')

    model_dict = torch.load("model_checkpoints/iemocap_4_best_dev_f1_model_atv.pt")
    
    stored_args = model_dict["args"]
    model = model_dict["state_dict"]
    testset = cogmen.Dataset(data["test"], stored_args)

    test = True
    with torch.no_grad():
        golds = []
        preds = []
        for idx in tqdm(range(len(testset)), desc="test" if test else "dev"):
            data = testset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(stored_args.device)
            y_hat = model(data)
            preds.append(y_hat.detach().to("cpu"))

        if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
        else:
            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
        print(test)
        if test:
            print(metrics.classification_report(golds, preds, digits=4))

            if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
                happy = metrics.f1_score(golds[:, 0], preds[:, 0], average="weighted")
                sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                anger = metrics.f1_score(golds[:, 2], preds[:, 2], average="weighted")
                surprise = metrics.f1_score(
                    golds[:, 3], preds[:, 3], average="weighted"
                )
                disgust = metrics.f1_score(golds[:, 4], preds[:, 4], average="weighted")
                fear = metrics.f1_score(golds[:, 5], preds[:, 5], average="weighted")

                f1 = {
                    "happy": happy,
                    "sad": sad,
                    "anger": anger,
                    "surprise": surprise,
                    "disgust": disgust,
                    "fear": fear,
                }
            print("WHY")
            print(f"F1 Score: {f1}")


if __name__ == "__main__":
    print("WHYYYYY")
    print("plz")
