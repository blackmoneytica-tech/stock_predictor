"""ML Stacker — 11 모듈 score를 feature로 actual_direction 학습.

목적: 가중치를 직접 정하지 않고 backtest 데이터로 자동 최적화.

Train data: backtest parquet files (v6/v7/v8)
Model: Logistic Regression (sklearn 없으면 numpy 직접 구현)
Features: 11개 module score + macro_breadth + days_to_earnings + post_catalyst_within
Target: actual_ret > 0 (bull) vs <= 0 (bear) → binary
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── 단순 logistic regression (sklearn 의존 없음) ────────────
class SimpleLogReg:
    """Newton-Raphson logistic regression."""

    def __init__(self, n_iter: int = 100, l2: float = 0.01):
        self.n_iter = n_iter
        self.l2 = l2
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.feature_names: List[str] = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        n, p = X.shape
        self.feature_names = feature_names
        # 표준화
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        Xs = (X - self.mean_) / self.std_
        # bias term 포함
        Xb = np.column_stack([np.ones(n), Xs])
        w = np.zeros(p + 1)
        for _ in range(self.n_iter):
            z = Xb @ w
            mu = self._sigmoid(z)
            grad = Xb.T @ (mu - y) + self.l2 * w
            S = mu * (1 - mu)
            H = Xb.T @ (Xb * S[:, None]) + self.l2 * np.eye(p + 1)
            try:
                w = w - np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        z = self.intercept_ + Xs @ self.coef_
        return self._sigmoid(z)


# ── 학습 ────────────────────────────────────────────────────
FEATURE_COLS_FROM_PARQUET = [
    "score",       # composite_score
    "confidence",
    "pred_ret_pct",
    # 추가 ML feature는 backtest parquet에 저장돼 있어야
]


def load_training_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """v6+v7+v8 backtest parquet 합쳐서 training set 빌드."""
    results_dir = Path(__file__).resolve().parents[2] / "data" / "results"
    dfs = []
    for name in ("volatile_daily.parquet", "v7_daily.parquet", "v8_daily.parquet"):
        p = results_dir / name
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise RuntimeError("No backtest data found")
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["actual_ret_pct"]).copy()

    # Features
    feat_cols = ["score", "confidence", "pred_ret_pct"]
    if "post_catalyst_within" in df.columns:
        df["has_catalyst"] = (df["post_catalyst_within"] <= 5).astype(float)
        feat_cols.append("has_catalyst")
    if "macro_mode" in df.columns:
        # one-hot for 모드
        for mode in ["BULL", "STRONG_BULL", "CHOPPY", "BEAR", "STRONG_BEAR"]:
            df[f"mode_{mode}"] = (df["macro_mode"] == mode).astype(float)
            feat_cols.append(f"mode_{mode}")
    if "beat_proxy" in df.columns:
        df["beat_proxy"] = df["beat_proxy"].fillna(0.5)
        feat_cols.append("beat_proxy")

    X = df[feat_cols].fillna(0).values.astype(float)
    y = (df["actual_ret_pct"] > 0).astype(float).values
    return X, y, feat_cols


def train_stacker() -> Dict:
    """학습 + 저장 + train accuracy 반환."""
    X, y, feat_cols = load_training_data()
    model = SimpleLogReg(n_iter=100, l2=0.05)
    model.fit(X, y, feat_cols)
    # train acc
    p = model.predict_proba(X)
    pred = (p > 0.5).astype(float)
    acc = float((pred == y).mean())
    # 저장
    with open(MODELS_DIR / "stacker.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": feat_cols, "train_acc": acc}, f)
    return {"train_acc": acc, "n_samples": len(y), "features": feat_cols}


# ── 예측 ────────────────────────────────────────────────────
_stacker_cache: Optional[Dict] = None


def _load_stacker():
    global _stacker_cache
    if _stacker_cache is not None:
        return _stacker_cache
    p = MODELS_DIR / "stacker.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        _stacker_cache = pickle.load(f)
    return _stacker_cache


def stacker_probability(
    composite_score: float,
    confidence: float,
    pred_ret_pct: float,
    post_catalyst_within: int = 999,
    macro_mode: str = "CHOPPY",
    beat_proxy: float = 0.5,
) -> Optional[float]:
    """학습된 stacker로 bull 확률 출력. 모델 없으면 None."""
    pkg = _load_stacker()
    if pkg is None:
        return None
    model = pkg["model"]
    feat_cols = pkg["feature_cols"]
    row = {
        "score": composite_score,
        "confidence": confidence,
        "pred_ret_pct": pred_ret_pct,
        "has_catalyst": 1.0 if post_catalyst_within <= 5 else 0.0,
        "mode_BULL": 1.0 if macro_mode == "BULL" else 0.0,
        "mode_STRONG_BULL": 1.0 if macro_mode == "STRONG_BULL" else 0.0,
        "mode_CHOPPY": 1.0 if macro_mode == "CHOPPY" else 0.0,
        "mode_BEAR": 1.0 if macro_mode == "BEAR" else 0.0,
        "mode_STRONG_BEAR": 1.0 if macro_mode == "STRONG_BEAR" else 0.0,
        "beat_proxy": beat_proxy,
    }
    x = np.array([[row.get(c, 0) for c in feat_cols]], dtype=float)
    p = model.predict_proba(x)[0]
    return float(p)
