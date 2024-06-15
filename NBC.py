import json
import random
from tqdm import tqdm
import numpy as np

from joblib import Parallel, delayed

class NBC:
    def __init__(self, name = None):
        self.name = name
        
        self.X_dir = {}
        self.prob_X_dir = {}
        self.X_count = 0
        
        self.Y_dir = {}
        self.prob_Y_dir = {}
        self.Y_count = 0
        
        self.dependent = {}
        self.dependent_prob = {}
        self.dependent_count = {}
    
    def fit(self, X, Y, progress_bar = True):
        assert len(X) == len(Y), 'The length of training examples must be the same'
        
        for i in tqdm(X, ascii = True, desc = 'Learn X prob. ', disable = not progress_bar):
            for j in i:
                if self.X_dir.get(j) is None:
                    self.X_dir[j] = 1
                else:
                    self.X_dir[j] += 1
                self.X_count += 1
                
        self.prob_X_dir = {i:self.X_dir[i]/self.X_count for i in self.X_dir}
        
        for i in tqdm(Y, ascii = True, desc = 'Learn Y prob. ', disable = not progress_bar):
            if self.Y_dir.get(i) is None:
                self.Y_dir[i] = 1
            else:
                self.Y_dir[i] += 1
            self.Y_count += 1
        self.prob_Y_dir = {i:self.Y_dir[i]/self.Y_count for i in self.Y_dir}
        
        for i in Y:
            if self.dependent.get(i) is None:
                self.dependent[i] = {}  
                self.dependent_count[i] = 0
                
        for x, y in tqdm(zip(X, Y), total = len(X), ascii = True, desc = 'Learn dependent prob. ', disable = not progress_bar):
            for i in x:
                if self.dependent[y].get(i) is None:
                    self.dependent[y][i] = 1
                else:
                    self.dependent[y][i] += 1
                self.dependent_count[y] += 1
                
        for i in self.dependent:
            sort_prob = sorted(self.dependent[i], reverse = True, key = lambda x: self.dependent[i][x])
            self.dependent[i] = {j:self.dependent[i][j] for j in sort_prob}
        
        self.dependent_prob = {i:{j:self.dependent[i][j]/self.dependent_count[i] for j in self.dependent[i]} for i in self.dependent}
    
    def predict(self, X, logit = False, progress_bar = True):
        predicts = []
        for i in tqdm(X, ascii = True, desc = 'Prediction', disable = not progress_bar):
            logits = {i:0 for i in self.prob_Y_dir}
            for k in logits:
                for j in i:
                    val = self.dependent_prob[k].get(str(j))
                    if val is None:
                        val = 1e-10
                    logits[k] += np.log(val)
                logits[k] += np.log(self.prob_Y_dir[k])
            
            if logit == False:
                max_logit = max(logits.values())
                label = [k for k, v in logits.items() if v == max_logit]
                if len(label) > 1:
                    label_prob = [self.prob_Y_dir[i] for i in label]
                    label = [l for l, p in zip(label, label_prob) if p == max(label_prob)][0]
                else:
                    label = label[0]
                    
                predicts += [label]
            else:
                predicts += [logits]
            
        return predicts
    
    def save(self, name):
        saved_params = json.dumps({
            'X_dir':self.X_dir,
            'prob_X_dir':self.prob_X_dir,
            'X_count':self.X_count,
            'Y_dir':self.Y_dir,
            'prob_Y_dir':self.prob_Y_dir,
            'Y_count':self.Y_count,
            'dependent':self.dependent,
            'dependent_prob':self.dependent_prob,
            'dependent_count':self.dependent_count,
        })
        
        with open(name, 'w') as f:
            f.write(saved_params)
            
    def load(self, name):
        with open(name, 'r') as f:
            saved_params = json.load(f)
            
        self.X_dir = saved_params['X_dir']
        self.prob_X_dir = saved_params['prob_X_dir']
        self.X_count = saved_params['X_count']
        self.Y_dir = saved_params['Y_dir']
        self.prob_Y_dir = saved_params['prob_Y_dir']
        self.Y_count = saved_params['Y_count']
        self.dependent = saved_params['dependent']
        self.dependent_prob = saved_params['dependent_prob']
        self.dependent_count = saved_params['dependent_count']
        
class NBCForest:
    def __init__(self, count, name = None):
        self.name = name
        
        self.count = count
        
        self.classifers = [NBC() for i in range(count)]
        
        self.Y_dir = {}
        self.prob_Y_dir = {}
        self.Y_count = 0
        
    def fit(self, X, Y, learn_coefs = False, progress_bar = True, seed = None):
        
        if seed is not None:
            np.random.seed(seed)
        
        for i in tqdm(Y, ascii = True, desc = 'Learn Y prob. ', disable = not progress_bar):
            if self.Y_dir.get(i) is None:
                self.Y_dir[i] = 1
            else:
                self.Y_dir[i] += 1
            self.Y_count += 1
        self.prob_Y_dir = {i:self.Y_dir[i]/self.Y_count for i in self.Y_dir}
        
        marks = np.random.randint(low = 0, high = self.count, size = len(X))
        for c, classifer in  tqdm(enumerate(self.classifers), total = self.count, ascii = True, desc = 'Learn classifers', disable = not progress_bar):
            
            X_ = [i for c_i, i in enumerate(X) if marks[c_i] == c]
            Y_ = [i for c_i, i in enumerate(Y) if marks[c_i] == c]
            
            classifer.fit(X_, Y_, progress_bar = False)
            
    def predict_worker(self, cls, X):
        return cls.predict(X, progress_bar = False)
    
    def predict(self, X, parallel = False, cores = 1, progress_bar = True):
        if parallel:
            labels = Parallel(n_jobs = cores)(delayed(self.predict_worker)(cls, X) for cls in self.classifers)
        else:
            labels = []
            for classifer in tqdm(self.classifers, ascii = True, desc = 'Calculate labels', disable = not progress_bar):
                labels += [classifer.predict(X, progress_bar = False)]
            
        votes = []
        for c_i, i in tqdm(enumerate(zip(*labels)), total = len(X), ascii = True, desc = 'Calculate votes', disable = not progress_bar):
            buf_votes = {}
            for c_j, j in enumerate(i):
                    
                if buf_votes.get(j) is None:
                    buf_votes[j] = 1
                else:
                    buf_votes[j] += 1
            
            max_vote = max(buf_votes.values())
            label = [k for k, v in buf_votes.items() if v == max_vote]
            
            if len(label) > 1:
                label = label[random.randint(0, len(label)-1)]
            else:
                label = label[0]
            
            votes += [label]

        return votes
    
    def save(self, name):
        saved_params = []
        for i in self.classifers:
            saved_params += [{
                'X_dir':i.X_dir,
                'prob_X_dir':i.prob_X_dir,
                'X_count':i.X_count,
                'Y_dir':i.Y_dir,
                'prob_Y_dir':i.prob_Y_dir,
                'Y_count':i.Y_count,
                'dependent':i.dependent,
                'dependent_prob':i.dependent_prob,
                'dependent_count':i.dependent_count,
            }]
            
        with open(name, 'w') as f:
            f.write(json.dumps(saved_params))
            
    def load(self, name):
        with open(name, 'r') as f:
            saved_params = json.load(f)
        
        for i, j in zip(self.classifers, saved_params):
            i.X_dir = j['X_dir']
            i.prob_X_dir = j['prob_X_dir']
            i.X_count = j['X_count']
            i.Y_dir = j['Y_dir']
            i.prob_Y_dir = j['prob_Y_dir']
            i.Y_count = j['Y_count']
            i.dependent = j['dependent']
            i.dependent_prob = j['dependent_prob']
            i.dependent_count = j['dependent_count']