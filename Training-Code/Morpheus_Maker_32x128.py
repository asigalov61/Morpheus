#!/usr/bin/env python
# coding: utf-8

# # Morpheus Maker (ver. 2.0)
# 
# ***
# 
# Powered by tegridy-tools TMIDIX Optimus Processors: https://github.com/asigalov61/tegridy-tools
# 
# ***
# 
# Credit for GPT2-RGA code used in this colab goes out @ Sashmark97 https://github.com/Sashmark97/midigen and @ Damon Gwinn https://github.com/gwinndr/MusicTransformer-Pytorch
# 
# ***
# 
# WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/
# 
# ***
# 
# #### Project Los Angeles
# 
# #### Tegridy Code 2022
# 
# ***

# # (Setup Environment)

# In[ ]:


#@title nvidia-smi gpu check
get_ipython().system('nvidia-smi')


# In[ ]:


#@title Install all dependencies (run only once per session)

get_ipython().system('git clone https://github.com/asigalov61/tegridy-tools')
get_ipython().system('pip install torch')
get_ipython().system('pip install tqdm')
get_ipython().system('pip install matplotlib')


# In[ ]:


#@title Import all needed modules

print('Loading needed modules. Please wait...')
import os
from datetime import datetime
import secrets
import copy
import tqdm as tqdm
from tqdm import tqdm

if not os.path.exists('/notebooks/Dataset'):
    os.makedirs('/notebooks/Dataset')

print('Loading TMIDIX module...')
os.chdir('/notebooks/tegridy-tools/tegridy-tools')
import TMIDIX

os.chdir('/notebooks/tegridy-tools/tegridy-tools')
from GPT2RGAX import *

import matplotlib.pyplot as plt

os.chdir('/notebooks/')


# # (FROM SCRATCH) Download and process MIDI dataset

# In[ ]:


#@title Download original LAKH/clean_midi MIDI subset (Recommended)

#@markdown Works best stand-alone/as-is for the optimal results
get_ipython().run_line_magic('cd', '/notebooks/')

get_ipython().system("wget 'http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz'")
get_ipython().system("tar -xvf 'clean_midi.tar.gz'")
get_ipython().system("rm 'clean_midi.tar.gz'")

get_ipython().run_line_magic('cd', '/notebooks/')


# In[ ]:


#@title Process MIDIs to special MIDI dataset with TMIDIX MIDI Processor

#@title Process MIDIs

sorted_or_random_file_loading_order = False # Sorted order is NOT recommended
dataset_ratio = 0.02 # Change this if you need more data


print('TMIDIX MIDI Processor')
print('Starting up...')
###########

files_count = 0

gfiles = []

melody_chords_f = []

###########

print('Loading MIDI files...')
print('This may take a while on a large dataset in particular.')

dataset_addr = "./clean_midi/"
# os.chdir(dataset_addr)
filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
print('=' * 70)

if filez == []:
    print('Could not find any MIDI files. Please check Dataset dir...')
    print('=' * 70)

if sorted_or_random_file_loading_order:
    print('Sorting files...')
    filez.sort()
    print('Done!')
    print('=' * 70)
else:
    print('Randomizing file list...')
    random.shuffle(filez)

    
stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print('Processing MIDI files. Please wait...')
for f in tqdm(filez[:int(len(filez) * dataset_ratio)]):
    try:
        fn = os.path.basename(f)
        fn1 = fn.split('.')[0]

        files_count += 1

        #print('Loading MIDI file...')
        score = TMIDIX.midi2ms_score(open(f, 'rb').read())

        events_matrix = []

        itrack = 1

        patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        patch_map = [[0, 1, 2, 3, 4, 5, 6, 7], # Piano 
                     [24, 25, 26, 27, 28, 29, 30], # Guitar
                     [32, 33, 34, 35, 36, 37, 38, 39], # Bass
                     [40, 41], # Violin
                     [42, 43], # Cello
                     [46], # Harp
                     [56, 57, 58, 59, 60], # Trumpet
                     [71, 72], # Clarinet
                     [73, 74, 75], # Flute
                     [-1], # Fake Drums
                     [52, 53] # Choir
                    ]

        while itrack < len(score):
            for event in score[itrack]:         
                if event[0] == 'note' or event[0] == 'patch_change':
                    events_matrix.append(event)
            itrack += 1

        events_matrix1 = []
        for event in events_matrix:
                if event[0] == 'patch_change':
                    patches[event[2]] = event[3]

                if event[0] == 'note':
                    event.extend([patches[event[3]]])
                    once = False
                    
                    for p in patch_map:
                        if event[6] in p and event[3] != 9: # Except the drums
                            event[3] = patch_map.index(p)
                            once = True
                            
                    if not once and event[3] != 9: # Except the drums
                        event[3] = 11 # All other instruments/patches channel
                        
                    if event[3] < 11: # We won't write all other instruments for now...
                        events_matrix1.append(event)
                        stats[event[3]] += 1

        events_matrix1.sort()

        #=======================

        if len(events_matrix1) > 0:
            events_matrix1.sort(key=lambda x: x[4], reverse=True)
            events_matrix1.sort(key=lambda x: (x[1], x[3]))

            cho = []
            pe = events_matrix1[0]
            melody_chords = []
            for e in events_matrix1:

                time = min(31, int(abs(e[1]-pe[1]) / 10))
                dur = min(127, int(abs(e[2] / 10)))
                cha = e[3]
                ptc = e[4]
                vel = e[5]

                melody_chords.append([time, dur, ptc, cha, vel])

                pe = e
            melody_chords_f.append(melody_chords)

        gfiles.append(f)

    except KeyboardInterrupt:
        print('Saving current progress and quitting...')
        break  

    except:
        print('Bad MIDI:', f)
        continue


# In[ ]:


# Process and mark INTs...

INTS_f1 = []

for chords_list in tqdm(melody_chords_f):
    INTS_f1.append([-1, -1, -1, -1, -1]) # Intro
    pe = chords_list[0]
    count = 0
    for i in chords_list:

        INTS_f1.append(i)
        
        if count == len(chords_list)-50:
            INTS_f1.append([-2, -2, -2, -2, -2]) # Outro
        
        count += 1
        pe = i
    INTS_f1.append([-3, -3, -3, -3, -3]) # End


# In[ ]:


INTS_f1[:15]


# In[ ]:


TMIDIX.Tegridy_Any_Pickle_File_Writer(INTS_f1, '/notebooks/Morpheus_INTS_32x128')


# In[ ]:


INTS_f1 = TMIDIX.Tegridy_Any_Pickle_File_Reader('/notebooks/Morpheus_INTS_32x128')


# In[ ]:


#@title Load processed INTs datasets
number_of_batches = 64 # Change this to your specs
n_workers = 30 # Change this to your specs
dataset_ratio = 0.25 # Change this if you want to limit input data
val_dataset_ratio = 0.03 # Change this if you want to limit input data

print('=' * 50)
print('Prepping INTs datasets...')


train_data1 = []

avg_vel = int(sum([y[4] for y in INTS_f1]) / len(INTS_f1))

pe = INTS_f1[0]

for i in tqdm(INTS_f1):

    if min(i) >= 0:
        
        if i[0] != 0:
            train_data1.extend([i[0] + int(i[1] * 32)])

        if i[4] > avg_vel: 
            train_data1.extend([(128 * 32) + 128 + (256 * i[3])+i[2]])
        else:
            train_data1.extend([(128 * 32) + (256 * i[3])+i[2]])

        pe = i
  
    if i == [-1, -1, -1, -1, -1]: # Intro
        train_data1.extend([(128 * 32)+(256 * 11)-3])

    if i == [-2, -2, -2, -2, -2]: # Outro
        train_data1.extend([(128 * 32)+(256 * 11)-2])

    if i == [-3, -3, -3, -3, -3]: # End
        train_data1.extend([(128 * 32)+(256 * 11)-1])

train_data = train_data1[:int(len(train_data1) * dataset_ratio)]

val_dataset = train_data[:int(len(train_data) * val_dataset_ratio)]
test_dataset = train_data[:int(len(train_data) * val_dataset_ratio)]

train_list = train_data
val_list = val_dataset
test_list = []
print('=' * 50)

print('Processing INTs datasets...')
train_dataset = EPianoDataset(train_list, max_seq, random_seq)
val_dataset = EPianoDataset(val_list, max_seq)
test_dataset = EPianoDataset(test_list, max_seq)
print('=' * 50)

print('Loading INTs datasets...')
batch_size = number_of_batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=n_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers)
print('=' * 50)

print('Total INTs in the dataset', len(train_data))
print('Total unique INTs in the dataset', len(set(train_data)))
print('Max INT in the dataset', max(train_data))
print('Min INT in the dataset', min(train_data))
print('=' * 50)

print('Checking datasets shapes...')
print('=' * 50)

print('Train loader')
for x, tgt in train_loader:
    print(f'X shape: {x.shape}')
    print(f'Target shape: {tgt.shape}')
    break
print('=' * 50)

print('Validation loader')
for x, tgt in val_loader:
    print(f'X shape: {x.shape}')
    print(f'Target shape: {tgt.shape}')
    break
print('=' * 50)

print('Test loader')
for x, tgt in test_loader:
    print(f'X shape: {x.shape}')
    print(f'Target shape: {tgt.shape}')
    break
print('=' * 50)

print('Done! Enjoy! :)')
print('=' * 50)


# # Test the resulting INTs dataset...

# In[ ]:


train_data[:15]


# In[ ]:


out = train_data[:16000]

if len(out) != 0:
    song = []
    song = out
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    duration = 0
    for s in song:
        if s >= 0 and s < 128 * 32:
            time += (s % 32) * 10
            dur = (s // 32) * 10

        if s >= 128 * 32 and s < (128 * 32) + (256 * 11):
            if (s // 128) % 2 != 0:
                vel = 90
                channel = ((s-128-(128 * 32)) // 256)
            else:
                vel = 60
                channel = ((s-(128 * 32)) // 256)

            pitch = s % 256

            song_f.append(['note', abs(time), dur, channel, pitch, vel ])

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Morpheus',  
                                                        output_file_name = '/notebooks/Morpheus-Music-Composition', 
                                                        track_name='Project Los Angeles', 
                                                        number_of_ticks_per_quarter=500)

    print('Done!')


# # (TRAIN)

# # Train the model

# In[ ]:


#@title Train

DIC_SIZE = max(train_data)+1

config = GPTConfig(DIC_SIZE, 
                   max_seq,
                   dim_feedforward=1024,
                   n_layer=8, 
                   n_head=8, 
                   n_embd=1024,
                   enable_rpr=True,
                   er_len=max_seq)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(config)

model = nn.DataParallel(model)

model.to(device)

#=====

init_step = 0
lr = LR_DEFAULT_START
lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)
eval_loss_func = nn.CrossEntropyLoss(ignore_index=DIC_SIZE)
train_loss_func = eval_loss_func

opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
lr_scheduler = LambdaLR(opt, lr_stepper.step)


#===

best_eval_acc        = 0.0
best_eval_acc_epoch  = -1
best_eval_loss       = float("inf")
best_eval_loss_epoch = -1
best_acc_file = '/notebooks/gpt2_rpr_acc.pth'
best_loss_file = '/notebooks/gpt2_rpr_loss.pth'
loss_train, loss_val, acc_val = [], [], []

for epoch in range(0, epochs):
    new_best = False
    
    loss = train(epoch+1, 
                 model, train_loader, 
                 train_loss_func, 
                 opt, 
                 lr_scheduler, 
                 num_iters=-1, 
                 save_checkpoint_steps=4000)
    
    loss_train.append(loss)
    
    eval_loss, eval_acc = eval_model(model, val_loader, eval_loss_func, num_iters=-1)
    loss_val.append(eval_loss)
    acc_val.append(eval_acc)
    
    if(eval_acc > best_eval_acc):
        best_eval_acc = eval_acc
        best_eval_acc_epoch  = epoch+1
        torch.save(model.state_dict(), best_acc_file)
        new_best = True

    if(eval_loss < best_eval_loss):
        best_eval_loss       = eval_loss
        best_eval_loss_epoch = epoch+1
        torch.save(model.state_dict(), best_loss_file)
        new_best = True
    
    if(new_best):
        print("Best eval acc epoch:", best_eval_acc_epoch)
        print("Best eval acc:", best_eval_acc)
        print("")
        print("Best eval loss epoch:", best_eval_loss_epoch)
        print("Best eval loss:", best_eval_loss)


# In[ ]:


# Eval funct to eval separately if needed

#=====

init_step = 0
lr = LR_DEFAULT_START
lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)
eval_loss_func = nn.CrossEntropyLoss(ignore_index=DIC_SIZE)
train_loss_func = eval_loss_func

opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
lr_scheduler = LambdaLR(opt, lr_stepper.step)


eval_loss, eval_acc = eval_model(model, val_loader, eval_loss_func, num_iters=-1)


# In[ ]:


#@title Plot resulting training loss graph

tr_loss_list = [item for sublist in loss_train for item in sublist]
plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
plt.savefig('/notebooks/Morpheus-Training-Loss-Graph.png')


# # (SAVE)

# In[ ]:


#@title Save the model

print('Saving the model...')
full_path_to_model_checkpoint = "/notebooks/Morpheus-Trained-Model.pth" #@param {type:"string"}
torch.save(model.state_dict(), full_path_to_model_checkpoint)
print('Done!')


# # Congrats! You did it! :)
