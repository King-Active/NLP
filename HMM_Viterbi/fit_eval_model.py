"""
    Created By Zeng zhiwen (zwzengi@Outlook.com)
    CQU NLP Experiment
"""

from collections import defaultdict
import re
import pickle
import gzip
from pathlib import Path
import yaml

class Pin2Word:
    def __init__(self, mode='train', save=False, save_path = None, load_path = None,
                 path_raw_txt=None, path_txt_pin=None, path_pin_2_han=None, **kwargs ):

        self.raw_txt_paths = path_raw_txt
        self.raw_txt_pin = path_txt_pin   
        self.raw_pin_2_han = open(path_pin_2_han, 'r', encoding='utf-8').read()
        self.valid_tokens = []
        self.invalid_token = {
            '1': re.compile(r'\d+'),          # 数字
            '2': re.compile(r'[a-zA-Z_]+'),   # 字母下划线
            '3': re.compile(r'[^\w]')         # 标点空格
        }
        self.hanzi_pinyin_pairs = []          # 存储(汉字, 拼音)对
        self.mode = mode  
        self.save = save
        self.save_path = save_path

        """ 模型参数 """     
        self.bi_counts = defaultdict(lambda: defaultdict(int))  # (汉字,汉字) 组合出现次数
        self.si_counts = defaultdict(int)                       # (汉字) 单字出现次数
        self.word_size = 0                                      # 非重复汉字个数
        self.total_cnt = 0                                      # 所有出现汉字个数
        self.pinyin_counts = defaultdict(int)                   # 拼音出现次数
        self.hanzi_pinyin_counts = defaultdict(int)             # (汉字,拼音)共现次数
        self.pin_2_han = defaultdict(str)                       # 拼音，汉字串

        if mode == 'train':
            assert path_raw_txt!=None and path_txt_pin!=None and path_pin_2_han!=None
            if save:
                assert save_path!=None
                
        if mode == 'infer':
            assert load_path != None
            self._load_model(load_path)
            print('Model load successfully!')

# private:

    def _save_model(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'bi_counts': {k: dict(v) for k, v in self.bi_counts.items()},
            'si_counts': dict(self.si_counts),
            'pinyin_counts': dict(self.pinyin_counts),
            'hanzi_pinyin_counts': dict(self.hanzi_pinyin_counts),
            "pin_2_han": dict(self.pin_2_han),
            'meta': {
                'word_size': self.word_size,
                'total_cnt': self.total_cnt
            }
        }

        with gzip.open(save_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

# private:

    def _load_model(self, load_path):
        with gzip.open(load_path, 'rb') as f:
            data = pickle.load(f)
            
            self.bi_counts = defaultdict(
                lambda: defaultdict(int))
            
            for k, v in data['bi_counts'].items():
                self.bi_counts[k].update(v)
            
            self.si_counts = defaultdict(int, data['si_counts'])
            self.pinyin_counts = defaultdict(int, data['pinyin_counts'])
            self.hanzi_pinyin_counts = defaultdict(int, data['hanzi_pinyin_counts'])
            self.pin_2_han = defaultdict(str, data['pin_2_han'])
            self.word_size = data['meta']['word_size']
            self.total_cnt = data['meta']['total_cnt']

    def _normalize(self, w):
        w = self.invalid_token['1'].sub('\x01', w)  
        w = self.invalid_token['2'].sub('\x01', w)
        w = self.invalid_token['3'].sub('\x01', w)
        return w
    
    def _norm_text(self, vis = False):
        """
        替换所有非汉字为 '\x01'
        连续非法字符合并为一个 '\x01'
        """
        res = ''
        for path in self.raw_txt_paths:
            raw_txt = open(path, 'r', encoding='utf-8').read()
            for t in raw_txt:
                res += self._normalize(t)

            res = re.sub(r'\x01+', '\x01', res)

            self.valid_tokens = list(res)

        if vis:
            print(repr(res))

    def _w_cnt(self, vis = False):
        """
        统计有效汉字单字频率
        有效汉字转移概率（若下一字符无效，也统计在内）
        """
        
        l = len(self.valid_tokens)

        for i in range(l):

            c_cur = self.valid_tokens[i]
            if i < l-1:
                c_nxt = self.valid_tokens[i+1]
            else:
                c_nxt = 'x01'

            if c_cur == '\x01':
                continue
            
            self.si_counts[c_cur] += 1
            
            self.bi_counts[c_cur][c_nxt] += 1

            self.word_size = len(self.si_counts)

        for _,val in self.si_counts.items():
            self.total_cnt += val

        if vis:
            print(self.si_counts)
            print(self.bi_counts)

    def _align_txt_pin(self, vis=False):
        """处理对齐文本"""
        for path in self.raw_txt_pin:
            raw_txt_pin = open(path, 'r', encoding='utf-8').read()

            lines = raw_txt_pin.strip().split('\n')
            for i in range(0, len(lines), 2):
                hanzi_line = lines[i].strip()
                pinyin_line = lines[i+1].strip()
                
                hanzi_chars = list(hanzi_line)
                pinyin_tokens = pinyin_line.split()
                
                for h, p in zip(hanzi_chars, pinyin_tokens):
                    if p != 'none':  # 跳过无拼音的符号
                        self.hanzi_pinyin_pairs.append((h, p))
                
                if vis:
                    print(self.hanzi_pinyin_pairs)

    def _count_pin_hanzi_bi(self, vis=False):
        """统计共现频率"""
        for h, p in self.hanzi_pinyin_pairs:
            self.pinyin_counts[p] += 1
            self.hanzi_pinyin_counts[(h, p)] += 1
        
        if vis:
            print(self.hanzi_pinyin_counts)

    def _pin_2_han(self, vis = False):
        lines = self.raw_pin_2_han.strip().split('\n')
        for l in lines:
            pinyin, hanzi = l.strip().split()
            self.pin_2_han[pinyin] = hanzi

        if vis:
            print(self.pin_2_han)

    def _get_pin_2_han(self, pin):
        """
        仅针对可能发射此拼音的汉字计算
        """
        return list(self.pin_2_han[pin])
    
    def _trans(self, i, j):
        """
        汉字转移概率 alpha_{ij}
        """

        # Laplace Smoothing
        return (self.bi_counts[i][j] + 1) /  (self.si_counts[i] + self.word_size)
    
    def _init(self, i):
        """
        汉字初始概率 pi_{i}
        每个字出现的频率 近似其 初始概率
        """
        return ( self.si_counts[i] + 1 ) / ( self.total_cnt + self.word_size)
        
    def _emis(self, i, j):
        if j not in self.pinyin_counts:
            return 1e-5
        
        total = self.pinyin_counts[j]  # 拼音出现几次
        count = self.hanzi_pinyin_counts.get((i, j), 0)    # 拼音和此汉字出现几次
        return (count + 1e-5) / (total + 1e-5 * len(self.hanzi_pinyin_counts))
    
# public:

    def hmm_train(self):
        
        assert self.mode == 'train'

        self._norm_text(False)
        self._w_cnt(False)
        self._align_txt_pin(False)
        self._count_pin_hanzi_bi(False)
        self._pin_2_han(False)

        if self.save:
            # 保存模型
            self._save_model(self.save_path)
            print(f"train finished and model saved successfully in {self.save_path}!")
        else:
            print(f"train finished!")

    def vtb_infer(self, pins):
        
        assert self.mode == 'infer'

        pin_list = pins.split()

        if not pin_list:
            return ""

        # --- 初始化状态 ---
        s = 0                   # 第一个有效拼音的位置
        invalid_prefix = ""     # 开头无效拼音段
        state = {}  
        while s < len(pin_list):

            pin = pin_list[s]
            han_list = self._get_pin_2_han(pin)

            if han_list:
                # 拼音有效
                for han in han_list:
                    state[han] = {
                        "prob": self._init(han) * self._emis(han, pin),
                        "path": invalid_prefix + han
                    }
                s += 1
                break

            else:
                invalid_prefix += r'\x01'
                s += 1
                

        for pin in pin_list[s:]:
            new_state = {}
            cur_han_list = self._get_pin_2_han(pin)
            
            # 当前拼音无候选汉字
            if not cur_han_list:

                # 保留状态
                for prev_han, prev_info in state.items():
                    new_state[prev_han] = {
                        "prob": prev_info["prob"] * 1e-5,
                        "path": prev_info["path"] + r'x\01'
                    }
                state = new_state
                continue

            # 正常处理流程
            for cur_han in cur_han_list:
                max_prob = -1.0
                best_path = ""

                for prev_han, prev_info in state.items():
                    prob = prev_info["prob"] * self._trans(prev_han, cur_han) * self._emis(cur_han, pin)
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_path = prev_info["path"] + cur_han
                
                if max_prob > 0:
                    new_state[cur_han] = {
                        "prob": max_prob,
                        "path": best_path
                    }
            
            state = new_state if new_state else state  # 无有效转移时保留原状态

        if not state:
            return f"无效拼音序列： {' '.join(pin_list)}"
        
        _, best_info = max( state.items(), 
                            key=lambda x: x[1]["prob"],
                            default=("<UNK>", {"path": ""}) )
    
        return str(best_info["path"])



if __name__ == '__main__':
    with open('config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        model = Pin2Word(**config)

        if config['mode'] == "train":
            model.hmm_train()

        else:
            correct_rate = 0.0

            with open (config['path_test'], 'r', encoding='utf-8') as f:
                text = f.read()
                lines = text.split('\n')
                for i in range(0, len(lines), 2):
                    pin = lines[i]
                    han = lines[i+1]

                    pin = pin.strip().lower()
                    pred = model.vtb_infer(pin)
                    print(f"{pin}: {repr(pred)}")

                    if len(han) != len(pred):
                        continue

                    cn = 0
                    for i in range(len(han)):
                        cn += 1 if han[i]==pred[i] else 0
                    correct_rate += cn/len(han)
                
                print(f"正确率：{correct_rate / (len(lines)/2)}")
                
