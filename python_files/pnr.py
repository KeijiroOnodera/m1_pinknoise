from typing import List
from sentence_transformers import SentenceTransformer
from neurodsp.aperiodic import compute_fluctuations
import numpy as np
import matplotlib.pyplot as plt
from my_chat_data import chat_history

class ChatInfo:
    def __init__(self, chat_id: int, speaker: str, chat_content: str):
        self.chat_id = chat_id
        self.speaker = speaker
        self.chat_content = chat_content

def embedding_chat_history(chat_history: List[ChatInfo]):
    model = SentenceTransformer('all-mpnet-base-v2')
    chat_contents = [chat.chat_content for chat in chat_history]
    return model.encode(chat_contents)

def calculate_semantic_synchrony(embedding_history: np.ndarray) -> np.ndarray:
    return np.array([
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        for a, b in zip(embedding_history[:-1], embedding_history[1:])
    ])

def calculate_PNR(chat_history: List[ChatInfo], n_shuffle: int = 1000) -> float:
    embedding_history = embedding_chat_history(chat_history)
    semantic_synchrony = calculate_semantic_synchrony(embedding_history)

    if len(semantic_synchrony) < 11:
        raise ValueError("PNRを計算するには最低12発話必要です")

    # 元のDFA
    original_array = semantic_synchrony.copy()
    _, _, original_scaling_coefficient = compute_fluctuations(
        original_array, fs=1.0, n_scales=20, min_scale=10, max_scale=len(original_array)
    )

    count = 0
    for _ in range(n_shuffle):
        shuffled = original_array.copy()
        np.random.shuffle(shuffled)
        _, _, shuffled_scaling = compute_fluctuations(
            shuffled, fs=1.0, n_scales=20, min_scale=10, max_scale=len(shuffled)
        )
        if shuffled_scaling < original_scaling_coefficient:
            count += 1

    return count / n_shuffle

#コサイン類似度の変遷をプロットする関数
def plot_semantic_synchrony(chat_history):
    embedding_history = embedding_chat_history(chat_history)
    synchrony = calculate_semantic_synchrony(embedding_history)
    plt.figure(figsize=(10, 4))
    plt.plot(synchrony, marker='o')
    plt.title("Semantic Synchrony Across Turns")
    plt.xlabel("Turn Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.show()

#PNRの変遷をプロットする関数
def plot_pnr_transition(chat_history: List[ChatInfo], n_shuffle: int = 100) -> None:
    pnr_scores = []
    turns = []

    for i in range(12, len(chat_history) + 1):  # 12ターン目からPNRが計算可能
        sub_history = chat_history[:i]
        try:
            pnr = calculate_PNR(sub_history, n_shuffle=n_shuffle)
            print("chat_id: {0}, PNR: {1}".format(i, pnr))
            pnr_scores.append(pnr)
            turns.append(i)
        except Exception as e:
            print(f"[ターン{i}] PNR計算エラー: {e}")
            continue

    # プロット
    plt.figure(figsize=(10, 4))
    plt.plot(turns, pnr_scores, marker='o')
    plt.title("PNR Transition Over Dialogue")
    plt.xlabel("Number of Turns")
    plt.ylabel("PNR Score")
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.show()


if __name__ == '__main__':    
    plot_semantic_synchrony(chat_history)
    plot_pnr_transition(chat_history)
    pnr = calculate_PNR(chat_history)
    print(f"PNR: {pnr:.4f}")

