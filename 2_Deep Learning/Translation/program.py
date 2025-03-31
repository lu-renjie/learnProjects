import torch
import os.path as osp
import torch.optim as optim
from datetime import datetime
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter

from loss import Loss
from data import Data
from utils import Logger, dict_to_markdown_table
from model import Seq2SeqAttention


class Program:
    def __init__(self, cfg):
        for key, value in cfg.items():
            setattr(self, key, value)
        self.device = torch.device(cfg['device'])
        self.log_dir = self.get_log_dir()

        self.data = Data(cfg)
        self.data.verbose()
        self.model = Seq2SeqAttention(
            cfg,
            word_vectors1=self.data.get_vectors_chinese(),
            word_vectors2=self.data.get_vectors_english(),
            tokens1=self.data.get_special_tokens_chinese(),
            tokens2=self.data.get_special_tokens_english())
        if cfg['load_trained_model']:
            state_dict = torch.load(self.trained_model_dir, map_location='cpu')
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        self.loss_fn = Loss(ignore_idx=self.data.get_special_tokens_english()['PAD'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)
        
        if self.log:
            self.writer = SummaryWriter(self.log_dir)
            self.logger = Logger(self.log_dir)
            self.writer.add_text('hyperparameters', dict_to_markdown_table(cfg))
        else:
            self.writer = None
            self.logger = Logger(None)

    def train(self):
        """
        Train machine translation model (Seq2Seq + Attention)
        Sentences are like ['<BOS>', 'i', 'love', 'you', '<EOS>', '<PAD>', '<PAD>']
        """
        best_epoch = None
        best_result = 0
        train_loader = self.data.get_train_loader()

        example = [
            '他一定是校長',
            '我会给你看图片',
            '我已经活了大半辈子了',
        ]

        for epoch in range(self.epoch_num):
            self.model.train()
            self.logger.log(f'EPOCH {epoch}')
            loss_sum = 0
            for i, (english, length_en, chinese, length_cn) in enumerate(train_loader):
                english = english.to(self.device)
                chinese = chinese.to(self.device)
                logits = self.model(chinese, length_cn, english, length_en)

                ground_truth = english[:, 1:]  # remove <BOS>
                loss = self.loss_fn(logits, ground_truth)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                self.logger.progress('TRAIN', f'loss: {loss_sum/(i+1):6.4f}', len(train_loader))
            self.scheduler.step()

            # test
            bleu = self.test()
            if bleu > best_result:
                best_epoch = epoch
                best_result = bleu
                if self.log:
                    torch.save(self.model.state_dict(), osp.join(self.log_dir, 'model.pth'))
                    self.logger.log('model saved')

            # log
            if self.writer is not None:
                self.writer.add_scalar('loss', loss_sum / (i+1), epoch)
                self.writer.add_scalar('bleu', bleu, epoch)

        for s in example:
            self.logger.log(s + ' --> ' + self.translate(s))
        self.logger.log('----------------finished training----------------')
        self.logger.log(f'best: {best_result}, best epoch: {best_epoch}')
        if self.log:
            torch.save(self.model.state_dict(), osp.join(self.log_dir, 'final_model.pth'))

    @torch.no_grad()
    def test(self):
        bleu_sum = 0
        test_set = self.data.get_test_set()

        self.model.eval()
        for i, (english, chinese) in enumerate(test_set):
            chinese = chinese.to(self.device)
            predicts = self.model.predict(chinese)
            predict_sentence = self.data.ids_to_words_english(predicts)

            english = english[1:-1]  # remove <BOS> and <EOS>
            reference = self.data.ids_to_words_english(english)

            bleu_sum += bleu_score([predict_sentence], [reference])
            self.logger.progress('TEST', f'bleu: {bleu_sum/(i+1):.4f}', total_step=len(test_set))
        return bleu_sum / (i+1)

    @torch.no_grad()
    def translate(self, sentence):
        ids = self.data.sentence_to_ids_chinese(sentence)
        ids = ids.to(self.device)
        predicts = self.model.predict(ids)
        predict_sentence = self.data.ids_to_words_english(predicts)
        predict_sentence = ' '.join(predict_sentence)
        return predict_sentence

    def get_log_dir(self):
        time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        dir1 = f'LSTM{self.num_layers}_{self.hidden_dim}'
        dir2 = f'{time}_{self.description}'
        log_dir = osp.join('log', dir1, dir2)
        return log_dir
