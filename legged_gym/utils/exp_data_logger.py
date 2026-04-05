from multiprocessing import Process, Value
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import itertools


class ExpLogger:
    def __init__(self, log_path, ref_key="base_lin_vel", length_limit=10000):
        self.state_log = {}
        self.output_path = log_path
        self.length_key = ref_key
        self.max_len = length_limit

        # check if the output directory exists
        output_dir = os.path.dirname(self.output_path)
        #     make it if necessary
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # check if the file already exists, if it does, increment a counter
        if os.path.exists(self.output_path):
            base, ext = os.path.splitext(self.output_path)  # handles .csv safely
            counter = 1
            new_path = f"{base}_{counter:02d}{ext}"
            while os.path.exists(new_path):
                counter += 1
                new_path = f"{base}_{counter:02d}{ext}"
            self.output_path = new_path




    def log_state(self, key, value):
        if key in self.state_log.keys():
            self.state_log[key].extend(value)
        else:
            self.state_log[key] = value

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

        if len(self.state_log[self.length_key]) > self.max_len:
            self.save_log()

    def save_log(self):
        print("Saving log contents to ", self.output_path)


        # Create DF and save it to CSV
        temp_df = pd.DataFrame.from_dict(self.state_log)
        temp_df.to_csv(self.output_path, mode='a', header=not os.path.exists(self.output_path))
        del temp_df
        self.reset()

    # def log_rewards(self, dict, num_episodes):
    #     for key, value in dict.items():
    #         if 'rew' in key:
    #             self.rew_log[key].append(value.item() * num_episodes)
    #     self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
