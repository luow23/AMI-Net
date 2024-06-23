import torch
from config import DefaultConfig

from models.AMI_Net_model import *
import numpy as np
opt = DefaultConfig()
print(opt.k)
model = eval(opt.model_name)(opt)

device = torch.device('cuda:0')

model.to(device)
dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions, 1))
model.eval()
#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input, 'test')
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     starter.record()
     _ = model(dummy_input, 'test')
     ender.record()
     # WAIT FOR GPU SYNC
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)
     timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn
print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
print(mean_syn)