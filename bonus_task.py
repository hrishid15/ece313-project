"""
ECE 313 - G  ▸  Bonus Task  (STUDENT STARTER CODE)
================================================================
    Fill in every "TODO" and run:
    $ python bonus_skeleton.py  --train-patients "1 2 ..." --test-patients "..."
"""
from __future__ import annotations
import argparse, pathlib, random, warnings
from typing import Tuple, Dict, List
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.io import loadmat
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



# ────────────────────────────────────────────────────────────────────────────
#                       ░ U T I L I T I E S  ░
# ────────────────────────────────────────────────────────────────────────────
def load_patient(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y)."""
    data = loadmat(mat_path)
    X = data["all_data"].astype(np.float32)
    y = data["all_labels"].ravel().astype(np.int64)
    return X, y


def list_patient_files(folder: str = ".") -> Dict[int, str]:
    """Map patient index → file path."""
    return {
        int(p.name.split("_")[0]): str(p)
        for p in pathlib.Path(folder).glob("*.mat")
    }


def splitTrainTest(patientFiles: Dict[int, str],
                   trainIds: List[int], testIds: List[int]
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trainXList, trainYList = [], []
    testXList, testYList = [], []

    for pid in trainIds:
        x, y = load_patient(patientFiles[pid])
        trainXList.append(x)
        trainYList.append(y)

    for pid in testIds:
        x, y = load_patient(patientFiles[pid])
        testXList.append(x)
        testYList.append(y)

    trainX = np.concatenate(trainXList, axis=1)
    trainY = np.concatenate(trainYList, axis=0)
    testX = np.concatenate(testXList, axis=1)
    testY = np.concatenate(testYList, axis=0)

    trainX = trainX.T
    testX = testX.T

    return trainX, trainY, testX, testY


def normalise(trainX: np.ndarray, testX: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(trainX, axis=0)
    std = np.std(trainX, axis=0)
    trainXNorm = (trainX - mean) / std
    testXNorm = (testX - mean) / std
    return trainXNorm, testXNorm


def empiricalPriors(yTrain: np.ndarray) -> Tuple[float, float]:
    pi0 = np.mean(yTrain == 0)
    pi1 = np.mean(yTrain == 1)
    return pi0, pi1


def metrics(yTrue: np.ndarray, yPred: np.ndarray) -> Dict[str, float]:
    falseAlarms = np.sum((yPred == 1) & (yTrue == 0))
    missDetections = np.sum((yPred == 0) & (yTrue == 1))
    total = len(yTrue)
    pFalseAlarm = falseAlarms / total
    pMissDetection = missDetections / total
    pError = 0.5 * (pFalseAlarm + pMissDetection)
    return {
        "P_false_alarm": pFalseAlarm,
        "P_miss_detection": pMissDetection,
        "P_error": pError
    }


# ────────────────────────────────────────────────────────────────────────────
#                ░  L O G I S T I C   R E G R E S S I O N  ░
# ────────────────────────────────────────────────────────────────────────────
def trainLogisticRegression(xTrain: np.ndarray, yTrain: np.ndarray):
    model = LogisticRegression(max_iter=500)
    model.fit(xTrain, yTrain)
    return model


# ────────────────────────────────────────────────────────────────────────────
#                  ░  F E E D - F O R W A R D   N N  ░
# ────────────────────────────────────────────────────────────────────────────
class FeedForwardNN(nn.Module):
    def __init__(self, d_in: int, d_h: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h),   nn.ReLU(),
            nn.Linear(d_h, 1),     nn.Sigmoid())
    def forward(self, x):          # (B,d) → (B,)
        return self.net(x).squeeze(1)


def train_nn(x_tr, y_tr, x_val, y_val,
             *, epochs=100, lr=1e-3, batch=256, patience=15, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FeedForwardNN(x_tr.shape[1]).to(dev).train()
    opt = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    best, best_val, wait = None, 1e9, 0
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.float().to(dev)
            opt.zero_grad(); loss_fn(net(xb), yb).backward(); opt.step()
        with torch.no_grad():
            v = loss_fn(net(torch.from_numpy(x_val).to(dev)),
                        torch.from_numpy(y_val).float().to(dev)).item()
        if v < best_val: best, best_val, wait = net.state_dict(), v, 0
        else:
            wait += 1
            if wait >= patience: break
    net.load_state_dict(best); net.eval().cpu(); return net


# ────────────────────────────────────────────────────────────────────────────
#                  ░       M A I N     ░
# ────────────────────────────────────────────────────────────────────────────
def main(args):
    patientFiles = list_patient_files(args.data_dir)
    trainIds = [int(i) for i in args.train_patients.split()]
    testIds = [int(i) for i in args.test_patients.split()]

    # 1) load & normalise
    trainX, trainY, testX, testY = splitTrainTest(patientFiles, trainIds, testIds)
    trainXNorm, testXNorm = normalise(trainX, testX)
    pi0, pi1 = empiricalPriors(trainY)
    tauML = 0.5
    tauMAP = pi0 / (pi0 + pi1)

    # 2) logistic regression
    logReg = trainLogisticRegression(trainXNorm, trainY)
    logRegScores = logReg.predict_proba(testXNorm)[:, 1]

    logRegPredsTauML = (logRegScores >= tauML).astype(int)
    logRegPredsTauMAP = (logRegScores >= tauMAP).astype(int)

    logRegMetricsTauML = metrics(testY, logRegPredsTauML)
    logRegMetricsTauMAP = metrics(testY, logRegPredsTauMAP)

    print("Logistic Regression:")
    print("  Threshold 0.5:", logRegMetricsTauML)
    print("  Threshold tauMAP:", logRegMetricsTauMAP)

    # 3) neural network (20% validation split)
    msk = np.random.rand(len(trainY)) < 0.8
    net = train_nn(trainXNorm[msk], trainY[msk], trainXNorm[~msk], trainY[~msk],
                   epochs=args.nn_epochs if hasattr(args, 'nn_epochs') else 100,
                   seed=args.seed)

    with torch.no_grad():
        nnScores = net(torch.from_numpy(testXNorm)).numpy()

    nnPredsTauML = (nnScores >= tauML).astype(int)
    nnPredsTauMAP = (nnScores >= tauMAP).astype(int)

    nnMetricsTauML = metrics(testY, nnPredsTauML)
    nnMetricsTauMAP = metrics(testY, nnPredsTauMAP)

    print("Neural Network:")
    print("  Threshold 0.5:", nnMetricsTauML)
    print("  Threshold tauMAP:", nnMetricsTauMAP)

    # 4) save to CSVs if needed (Bonus Task 2)
    # (optional, not required unless specified)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--train-patients", required=True,
                    help="e.g. '1 2 4 5 6 7'")
    ap.add_argument("--test-patients",  required=True,
                    help="e.g. '3 8 9'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
