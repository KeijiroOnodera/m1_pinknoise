from typing import List
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

from neurodsp.sim import sim_powerlaw
import numpy as np 
# Import the function for computing fluctuation analyses
from neurodsp.aperiodic import compute_fluctuations
from numpy.typing import NDArray

from my_chat_data import chat_history

class chat_info:
    def __init__(self, chat_id: int, speaker: str, chat_content: str):
        self.chat_id = chat_id
        self.speaker = speaker
        self.chat_content = chat_content

def embedding_chat_history(chat_hisotry: List[chat_info]) -> NDArray:
    model = SentenceTransformer('all-mpnet-base-v2')
    # トークナイザーを取得
    tokenizer = model.tokenizer

    # truncate_side を変更
    tokenizer.truncation_side = "left"  # "left" または "right" を指定可能

    chat_contents = [chat.chat_content for chat in chat_hisotry]
    chat_embedding = model.encode(chat_contents)
    return chat_embedding

def calculate_semantic_synchrony(embedding_hisotry: NDArray) -> NDArray:
    # semantic_synchrony[i]は、chat_id: iの会話とchat_id: i+1の会話の類似度を表す
    semantic_synchrony = []
    for i in range(len(embedding_hisotry)):
        if i + 1 >= len(embedding_hisotry):
            break
        a = embedding_hisotry[i]
        b = embedding_hisotry[i+1]
        cos = np.dot(a, b)/ (np.linalg.norm(a) * np.linalg.norm(b))
        semantic_synchrony.append(cos)
    return np.array(semantic_synchrony)

    
def calculate_PNR(chat_hisotry: List[chat_info]):
    embedding_history = embedding_chat_history(chat_hisotry)
    semantic_synchrony = calculate_semantic_synchrony(embedding_history)
    ts_pl, flucs_pl, exp_pl = compute_fluctuations(semantic_synchrony, 1, n_scales=20,
                                                min_scale=10, max_scale=len(semantic_synchrony))
    original_scaling_coefficient = exp_pl
    original_array = semantic_synchrony.copy()
    count_white_noise = 0
    for i in range(1000):
        np.random.shuffle(semantic_synchrony)
        ts_pl, flucs_pl, exp_pl = compute_fluctuations(semantic_synchrony, 1, n_scales=20,
                                                min_scale=10, max_scale=len(semantic_synchrony))
        if exp_pl < original_scaling_coefficient:
            count_white_noise += 1
    return (count_white_noise/1000)
#print(calculate_PNR(chat_hisotry))


# ピンクノイズを生成して、スケーリング係数が1になるか確認。
if __name__ == '__main__':
    # パラメータ
    duration = 5  # 秒
    sampling_rate = 1000  # Hz

    # ピンクノイズを生成
    pink_noise = sig_pl = sim_powerlaw(100, 1, exponent=-1)
    ts_pl, flucs_pl, exp_pl = compute_fluctuations(pink_noise, 1, n_scales=20,
                                                min_scale=10, max_scale=len(pink_noise))
    original_scaling_coefficient = exp_pl
    print("original_scaling_coefficient", original_scaling_coefficient)
    # original_scaling_coefficient 1.0890747935404612(ランダムで変わる)

    # chat_hisotry: List[chat_info] = [chat_info(1, 'Alice', 'Hello'), chat_info(2, 'Bob', 'Hi'), chat_info(3, 'Alice', 'How are you?'), chat_info(4, 'Bob', 'I am fine.'), chat_info(5, 'Alice', 'Good to hear that.'), chat_info(6, 'Bob', 'How about you?'), chat_info(7, 'Alice', 'I am good too.'), chat_info(8, 'Bob', 'That is great.'), chat_info(9, 'Alice', 'Bye'), chat_info(10, 'Bob', 'See you later.'), chat_info(11, 'Alice', 'Goodbye'), chat_info(12, 'Bob', 'Goodbye'), chat_info(13, 'Alice', 'Take care'), chat_info(14, 'Bob', 'You too'), chat_info(15, 'Alice', 'Have a nice day'), chat_info(16, 'Bob', 'You too'), chat_info(17, 'Alice', 'See you tomorrow'), chat_info(18, 'Bob', 'See you tomorrow')]
    # 計算効率は悪いが、この方法が良かった。
    #最小会話数は12。会話数が11の時は、類似度信号の個数が11-1=10個になり、ウィンドウサイズが10~10で20分割するためワーニングが出る。それ以下だと、10~10未満で20分割するという意味のわからないこととなるため、エラーが発生。
    #0~i-1までの会話数でPNRを計算
    for i in range(12, len(chat_history)+1):
        pnr = calculate_PNR(chat_history[:i])
        print("chat_id: {0}, PNR: {1}".format(i-1, pnr))

