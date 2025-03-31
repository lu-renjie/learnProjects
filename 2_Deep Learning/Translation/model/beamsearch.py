import torch


def matrix_topk(matrix, k):
    """
    Args:
        matrix: (m, n) tensor
        k: top-k
    Returns:
        values: (k,) tensor, 最高的k个值
        result: (k, 2) tensor, 每一行表示值最高的元素的2d索引
    """
    m, n = matrix.shape
    result = torch.zeros(k, 2, dtype=torch.long)

    values, indices = matrix.flatten().topk(k)
    result[:, 0] = torch.div(indices, n, rounding_mode='floor')
    result[:, 1] = indices % n
    return values, result


class BeamSearch:
    def __init__(self, k, EOS, max_length):
        self.k = k                    # beam search的k
        self.EOS = EOS                # 用于判断一个序列是否结束
        self.max_length = max_length  # 序列的最大搜索长度

        self.length = 0        # 当前序列的长度
        self.finished = False  # 表示beam search是否搜索结束
        self.scores = []       # 记录k个未结束序列的分数
        self.sequences = []    # 记录k个未结束的序列
        self.finished_scores = []       # 记录搜索结束的序列的分数
        self.finished_sequences = []     # 记录搜索结束的序列

    def __check_finished(self):
        """
        检查是否结束
        达到了最大长度, 或所有序列都预测到了EOS就算结束
        """
        if self.k == 0 or self.length == self.max_length:
            self.finished = True
            self.finished_scores += self.scores
            self.finished_sequences += self.sequences

    def first(self, scores):
        self.length += 1
        scores = scores.detach().cpu().squeeze(0)  # (C,)
        values, indices = scores.topk(self.k)
        values, indices = values.numpy(), indices.numpy()

        selection_next = []
        selection_former = []
        for i in range(self.k):
            choice = indices[i]
            if choice == self.EOS:
                self.k -= 1
                self.finished_sequences.append([])
            else:
                self.scores.append(values[i])
                self.sequences.append([choice])
                selection_next.append(choice)
                selection_former.append(0)

        self.__check_finished()
        return selection_former, selection_next

    def step(self, new_scores):
        """
        输入上一步decoder预测的每个词的分数, 根据Beam Search搜索规则返回下一步应该选择哪些token

        Args:
            new_scores: (k, C) tensor
                k表示当前序列的个数, beam search最多有k个序列, 有些序列预测到EOS就结束了, 此时k
                减小.
                C表示词的个数
                new_socores表示k各序列预测每个词的概率的对数
        Returns:
            selection_former: list[int], 表示选择哪些序列保留
            selection_next: list[int], 表示下一步应该选哪些词
        """
        new_scores = new_scores.detach().cpu()

        self.length += 1
        # 计算分数
        old_scores = torch.tensor(self.scores).unsqueeze(1)  # (k, 1)
        scores = old_scores * (self.length - 1) + new_scores
        scores = scores / self.length

        # 选择最好的k个序列
        selection_former = []
        selection_next = []
        new_scores = []
        new_predicts = []

        values, indices = matrix_topk(scores, self.k)
        values, indices = values.numpy(), indices.numpy()
        for i in range(self.k):  # 遍历每个序列
            m, n = indices[i, :]  # m表示先前的哪个序列, n表示选择哪个单词
            score = values[i]
            if n == self.EOS:  # 预测到EOS, 该序列结束
                self.k -= 1
                self.finished_sequences.append(self.sequences[m])
                self.finished_scores.append(score)
                continue
            new_predicts.append(self.sequences[m] + [n])
            new_scores.append(score)
            selection_former.append(m)
            selection_next.append(n)
        self.scores = new_scores
        self.sequences = new_predicts

        # 检查是否结束, 并返回
        self.__check_finished()
        return selection_former, selection_next

    def get_best_predict(self):
        best_score = torch.tensor(self.finished_scores)
        idx = best_score.argmax()
        return self.finished_sequences[idx]

    def not_finished(self):
        return not self.finished
