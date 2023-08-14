from pathlib import Path
import numpy as np
import pandas as pd
import os
import json
from torch.utils.tensorboard import SummaryWriter

class AutoLogger():
    def __init__(self, metrics, phases=['train', 'test'], tensorboard=False, 
                 save_dir=None, tb_dir=None):
        
        self.epoch_values = dict()
        self.batch_values = dict()
        self.temp_values  = dict()
        self.phases = phases
        self.metrics = metrics
        self.tb = tensorboard
        self.save_dir = save_dir
        self.tb_dir = tb_dir
        self.meta_template = ['MODEL TYPE', 
                              'SAVE DIR', 
                              'DATETIME', 
                              'HYPERPARAMS', 
                              'EXP COMMENTS/DESC/DETAILS']
        if self.save_dir is None:
            self.save_dir = Path('./logs')
        else:
            self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True)
        current_count = len(os.listdir(self.save_dir))+1
        self.save_dir = self.save_dir/str(current_count)
        self.save_dir.mkdir(exist_ok=True)
            
        
        if self.tb and (self.tb_dir is None):
            self.tb_dir = Path('./tb_logs')
        elif self.tb:
            self.tb_dir = Path(self.tb_dir)
        
        if self.tb:
            self.tb_dir.mkdir(exist_ok=True)
            current_count = len(os.listdir(self.tb_dir))+1
            self.tb_dir = self.tb_dir/str(current_count)
            self.tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(self.tb_dir)

        for phase in self.phases:
            self.epoch_values[phase] = dict()
            self.batch_values[phase] = dict()
            # self.temp_values[phase] = dict()

            self.epoch_values[phase]['loss'] = list()
            self.batch_values[phase]['loss'] = list()
            # self.temp_values[phase] = list()

            for metric, fn in self.metrics.items():
                self.epoch_values[phase][metric] = list()
                self.batch_values[phase][metric] = list()

    def add_metadata(self, metadict):
        with open(self.save_dir/'meta.json') as f:
            json.dump(metadict, f)
    
    # def save_files(self):
    #     pass

    def epoch_start(self, epoch=None):

        for phase in self.phases:
            self.temp_values[phase] = dict()
            self.temp_values[phase]['loss'] = list()

            for metric, fn in self.metrics.items():
                self.temp_values[phase][metric] = list()
                
    def epoch_end(self, epoch=None, model=None):
        for phase in self.phases:
            for metric, fn in self.metrics.items():
                values = self.temp_values[phase][metric]
                self.batch_values[phase][metric].extend(values)
                self.epoch_values[phase][metric].append(np.array(values).mean())
                ##add to tensorboard
                if self.tb:
                    self.writer.add_scalar(f"{phase}/epoch/{metric}", 
                                        np.array(values).mean(),
                                        len(self.epoch_values[phase][metric])-1)
                
            loss_vals = self.temp_values[phase]['loss']
            self.batch_values[phase]['loss'].extend(loss_vals)
            self.epoch_values[phase]['loss'].append(np.array(loss_vals).mean())
            if self.tb:
                    self.writer.add_scalar(f"{phase}/epoch/loss", 
                                        np.array(loss_vals).mean(),
                                        len(self.epoch_values[phase]['loss'])-1)
            ##add to tensorboard

            ##save dicts
            pd.DataFrame(self.batch_values[phase]).to_csv(
                                self.save_dir/(phase+' batch metrics.csv'),
                                index=False)
            pd.DataFrame(self.epoch_values[phase]).to_csv(
                                self.save_dir/(phase+' epoch metrics.csv'),
                                index=False)
            
            

    def log_loss(self, loss_value, phase):
        self.temp_values[phase]['loss'].append(loss_value)
        if self.tb:
            self.writer.add_scalar(f"{phase}/batch/loss", loss_value, 
                len(self.batch_values[phase]['loss'])+len(self.temp_values[phase]['loss']) -1)
            
    
    def log_metrics(self, true_value, predicted, phase):
        for metric, fn in self.metrics.items():
            self.temp_values[phase][metric].append(fn(true_value, predicted))
            if self.tb:
                self.writer.add_scalar(f"{phase}/batch/{metric}", 
                        fn(true_value, predicted),
                        len(self.batch_values[phase]['loss'])+len(self.temp_values[phase]['loss']) -1)
