from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import time
import random
import os
import numpy as np

from Engine import MCTSForest, MCTSTickTacToe, Tournament
from Model import PolicyNet

BUFFER_SIZE = 500000
BATCH_SIZE = 512
NUM_PRODUCERS = 6
NUM_ITERS = 1000
GAMES_PER_PRODUCER = 256
NORMAL_GAMES_NUM = 32
TOUNAMENT_GAMES = 25
SAVE_PATH = "checkpoints7/"
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class SharedReplayBuffer:
    def __init__(self, capacity, model:nn.Module):
        self.capacity = capacity
        self.data_tensor = torch.zeros((capacity, 3,9,9)).share_memory_()
        self.target_tensor = torch.zeros((capacity, 1)).share_memory_()
        self.prop_tensor = torch.zeros((capacity,9,9)).share_memory_()
        self.pos = mp.Value('i', 0)
        self.size = mp.Value('i', 0)
        manager = mp.Manager()
        self.lock = mp.Lock()
        self.best_dict = manager.dict()
        self.model_lock = mp.Lock()
        self.version = mp.Value('i', 0)
        self.replace_best_model(model)
        self.training_iters = mp.Value('i', 0)
    def add(self, data, target, prop):
        with self.lock:
            idx = self.pos.value
            self.data_tensor[idx] = data
            self.target_tensor[idx] = target
            self.prop_tensor[idx] = prop
            self.pos.value = (self.pos.value + 1) % self.capacity
            if self.size.value < self.capacity:
                self.size.value += 1

    def sample(self, batch_size):
        with self.lock:
            current_size = self.size.value

        if current_size < batch_size:
            return None, None, None

        indices = torch.randint(0, current_size, (batch_size,))
        return self.data_tensor[indices], self.target_tensor[indices], self.prop_tensor[indices]
    def get_best_model_ver(self):
        with self.model_lock:
            return self.version.value
    def best_model(self):
        with self.model_lock:
            return {k: v.clone() for k, v in self.best_dict.items()}
    def replace_best_model(self, model:nn.Module):
        with self.model_lock:
            self.best_dict.clear()
            for k,v in model.state_dict().items():
                self.best_dict[k] = v
            self.version.value += 1
    def bump_iter(self):
        with self.lock:
            self.training_iters.value += 1
def producer_process(rank, buffer: SharedReplayBuffer, stop_event):
    pid = os.getpid()
    print(f"Producer {rank} (PID: {pid}) started.")
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    global_counter = 0
    model = PolicyNet(num_blocks=8).to(DEVICE)
    version = -1
    compiled_model = torch.jit.trace(model, torch.zeros((1, 3, 9, 9)).to(DEVICE))
    while not stop_event.is_set():
        if version < buffer.get_best_model_ver():
            version = buffer.get_best_model_ver()
            model.load_state_dict(buffer.best_model())
            compiled_model = torch.jit.trace(model, torch.zeros((1, 3, 9, 9)).to(DEVICE))
        forest = MCTSForest(compiled_model,DEVICE,shards=1)
        training_temp = 2
        if buffer.training_iters.value < 10:
            cpuct = 2
        else:
            cpuct = 1
        if buffer.training_iters.value < 8 and buffer.training_iters.value > 0:
            training_temp /= buffer.training_iters.value / 4
        elif buffer.training_iters.value > 8:
            training_temp = 1
        for _ in range(START_GAMES if global_counter ==0  else NORMAL_GAMES_NUM):
            forest.add_game(MCTSTickTacToe(compiled_model,DEVICE,forest,training_mode=True, epsilon=0.25))
        local_counter = 1
        while not forest.are_all_finished():
            forest.execute_iterations(NUM_ITERS)
            forest.step_forward(training_temp if local_counter < 30 else 0)
            print(f"Producer {rank} executed move {local_counter}")
            local_counter += 1
        commited = 0
        for game in forest.games:
            samples = game.get_training_data()
            for sample in samples:
                buffer.add(sample[0], sample[2], sample[1])
                commited += 1
        print(f"Producer {rank} comitted new games({commited} new moves)")
        global_counter += 1



def consumer_process(buffer:SharedReplayBuffer, stop_event):
    print("Learner started.")
    model = PolicyNet(num_blocks=8)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

    step = 0
    value_loss_sum, policy_loss_sum = 0, 0
    while not stop_event.is_set():
        data, target, prop_target = buffer.sample(BATCH_SIZE)

        if data is None:
            time.sleep(10)  # Wait for buffer to fill
            continue
        if step%5000 == 0 and step > 0:
            checkpoint = {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, f"{SAVE_PATH}step_{step}.pt")
            oldmodel = PolicyNet(num_blocks=8)
            oldmodel.load_state_dict(buffer.best_model())
            oldmodel_traced = torch.jit.trace(oldmodel, torch.randn(1,3,9,9)).to(DEVICE)
            newmodel_traced = torch.jit.trace(deepcopy(model), torch.randn(1,3,9,9)).to(DEVICE)
            newmodel_traced.eval()
            oldmodel_traced.eval()
            tournament = Tournament(newmodel_traced,oldmodel_traced,DEVICE)
            result = tournament.play(TOUNAMENT_GAMES,NUM_ITERS)
            result = np.array(result)
            wins = np.sum(result==1)
            print(f"Tournament results {wins} / {TOUNAMENT_GAMES*2} (lost: {np.sum(result==-1)}): {wins/(TOUNAMENT_GAMES*2)}")
            if wins/(TOUNAMENT_GAMES*2) >= 0.55:
                buffer.replace_best_model(model)
                buffer.bump_iter()
            step += 1
            continue


        optimizer.zero_grad()
        p, v = model(data)
        p = p.reshape(-1,9,9)
        p[data[:,2,:,:]==0] = -1e9
        log_probs = F.log_softmax(p.reshape(-1,81),dim=1)
        prop_target = prop_target.reshape(-1,81)
        value_loss, policy_loss = F.mse_loss(target, v), -(prop_target*log_probs).sum(dim=1).mean()
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()
        value_loss_sum += value_loss
        policy_loss_sum += policy_loss
        step += 1
        if step % 100 == 0:
            print(f"[Learner] Step {step} | Value Loss: {value_loss_sum / 100} | Policy Loss: {policy_loss_sum / 100} | Buffer Size: {buffer.size.value}")
            value_loss_sum, policy_loss_sum = 0, 0


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    model = PolicyNet(num_blocks=8)
    shared_buffer = SharedReplayBuffer(BUFFER_SIZE,model)
    stop_event = mp.Event()

    processes = []

    p_learner = mp.Process(target=consumer_process, args=(shared_buffer, stop_event))
    p_learner.start()
    processes.append(p_learner)

    for rank in range(NUM_PRODUCERS):
        p = mp.Process(target=producer_process, args=(rank, shared_buffer, stop_event))
        p.start()
        processes.append(p)

    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        data = {
            "games": shared_buffer.data_tensor,
            "values": shared_buffer.target_tensor,
            "props": shared_buffer.prop_tensor,
            "size": shared_buffer.size.value
        }
        torch.save(data, SAVE_PATH+"training_data.pt")
        stop_event.set()
        for p in processes:
            p.join()
        print("All processes stopped.")
