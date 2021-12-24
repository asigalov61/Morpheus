#!/usr/bin/env python
# coding: utf-8

# # Morpheus Maker (ver. 1.0)
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
# #### Tegridy Code 2021
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


#@title Download Multi-Instrumental European MuseNet MIDI dataset (Recommended)

#@markdown Original Morpheus Model Dataset

#@markdown Works best stand-alone/as-is for the optimal results
get_ipython().run_line_magic('cd', '/notebooks/Dataset/')

get_ipython().system("wget 'https://github.com/asigalov61/Tegridy-MIDI-Dataset/raw/master/World-MuseNet-MIDI-Dataset/European-MuseNet.zip'")
get_ipython().system("unzip -j '/notebooks/Dataset/World-MuseNet-MIDI-Dataset/European-MuseNet.zip'")
get_ipython().system("rm '/notebooks/Dataset/World-MuseNet-MIDI-Dataset/European-MuseNet.zip'")

get_ipython().run_line_magic('cd', '/notebooks/')


# In[ ]:


#@title Process MIDIs to special MIDI dataset with Tegridy MIDI Processor

#@markdown IMPORTANT NOTES:

#@markdown 1) Best results are achieved with the single-track, single-channel, single-instrument MIDI 0 files with plain English names (avoid special or sys/foreign chars)

#@markdown 2) MIDI Channel = -1 means all MIDI channels except the drums. MIDI Channel = 16 means all channels will be processed. Otherwise, only single indicated MIDI channel will be processed

desired_dataset_name = "Morpheus-Music-Dataset" #@param {type:"string"}
file_name_to_output_dataset_to = "/notebooks/Morpheus-Music-Dataset" #@param {type:"string"}
desired_MIDI_channel_to_process = 16 #@param {type:"slider", min:-1, max:16, step:1}
sorted_or_random_file_loading_order = False #@param {type:"boolean"}
encode_velocities = True #@param {type:"boolean"}
encode_MIDI_channels = True #@param {type:"boolean"}
add_transposed_dataset_by_this_many_pitches = 0 #@param {type:"slider", min:-12, max:12, step:1}
add_transposed_and_flipped_dataset = False #@param {type:"boolean"}
chordify_input_MIDIs = False #@param {type:"boolean"}
melody_conditioned_chords = False #@param {type:"boolean"}
melody_pitch_baseline = 60 #@param {type:"slider", min:0, max:127, step:1}
time_denominator = 1 #@param {type:"slider", min:1, max:50, step:1}
transform_to_pitch = 0 #@param {type:"slider", min:0, max:127, step:1}
perfect_timings = True #@param {type:"boolean"}
MuseNet_encoding = True #@param {type:"boolean"}
chars_encoding_offset = 0 #@param {type:"number"}

print('TMIDI Optimus MIDI Processor')
print('Starting up...')
###########

average_note_pitch = 0
min_note = 127
max_note = 0

files_count = 0

gfiles = 0

chords_list_f = []
melody_list_f = []

chords_list = []
chords_count = 0

melody_chords = []
melody_count = 0

TXT_String = ''

TXT = ''
melody = []
chords = []
INTS_f = []

flist = []

###########

print('Loading MIDI files...')
print('This may take a while on a large dataset in particular.')

dataset_addr = "/notebooks/Dataset/"
os.chdir(dataset_addr)
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
  random.shuffle(filez)

# Stamping the dataset info
print('Stamping the dataset info...')

TXT_String += 'DATASET=' + str(desired_dataset_name) + chr(10)
TXT_String += 'CREATED_ON=' + str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-') + chr(10)

TXT_String += 'CHARS_ENCODING_OFFSET=' + str(chars_encoding_offset) + chr(10)
TXT_String += 'TIME_DENOMINATOR=' + str(time_denominator) + chr(10)
TXT_String += 'TRANSFORM=' + str(transform_to_pitch) + chr(10)
TXT_String += 'PERFECT_TIMINGS=' + str(perfect_timings) + chr(10)
TXT_String += 'MUSENET_ENCODING=' + str(MuseNet_encoding) + chr(10)
TXT_String += 'TRANSPOSED_BY=' + str(add_transposed_dataset_by_this_many_pitches) + chr(10)
TXT_String += 'TRANSPOSED_AND_FLIPPED=' + str(add_transposed_and_flipped_dataset) + chr(10)

TXT_String += 'LEGEND=STA-DUR-PTC'
if encode_velocities:
  TXT_String += '-VEL'
if encode_MIDI_channels:
  TXT_String += '-CHA'
TXT_String += chr(10)

print('Processing MIDI files. Please wait...')
for f in tqdm(filez):
  try:
    fn = os.path.basename(f)
    fn1 = fn.split('.')[0]

    files_count += 1
    TXT, melody, chords, bass_melody, karaokez, INTS, aux1, aux2 = TMIDIX.Optimus_MIDI_TXT_Processor(f, chordify_TXT=chordify_input_MIDIs, output_MIDI_channels=encode_MIDI_channels, char_offset=chars_encoding_offset, dataset_MIDI_events_time_denominator=time_denominator, output_velocity=encode_velocities, MIDI_channel=desired_MIDI_channel_to_process, MIDI_patch=range(0, 127), melody_conditioned_encoding=melody_conditioned_chords, melody_pitch_baseline=melody_pitch_baseline, perfect_timings=perfect_timings, musenet_encoding=MuseNet_encoding, transform=transform_to_pitch)
    TXT_String += TXT
    melody_list_f += melody
    chords_list_f.append(chords)
    INTS_f.append(INTS)
    flist.append([f, fn1])
    gfiles += 1

    if add_transposed_dataset_by_this_many_pitches != 0:

      TXT, melody, chords, bass_melody, karaokez, INTS, aux1, aux2 = TMIDIX.Optimus_MIDI_TXT_Processor(f, chordify_TXT=chordify_input_MIDIs, output_MIDI_channels=encode_MIDI_channels, char_offset=chars_encoding_offset, dataset_MIDI_events_time_denominator=time_denominator, output_velocity=encode_velocities, MIDI_channel=desired_MIDI_channel_to_process, transpose_by=add_transposed_dataset_by_this_many_pitches, MIDI_patch=range(0, 127), melody_conditioned_encoding=melody_conditioned_chords, melody_pitch_baseline=melody_pitch_baseline, perfect_timings=perfect_timings, musenet_encoding=MuseNet_encoding, transform=transform_to_pitch)
      TXT_String += TXT
      melody_list_f += melody
      chords_list_f.append(chords)
      INTS_f.append(INTS)
      gfiles += 1

    if add_transposed_and_flipped_dataset == True:

      TXT, melody, chords, bass_melody, karaokez, INTS, aux1, aux2 = TMIDIX.Optimus_MIDI_TXT_Processor(f, chordify_TXT=chordify_input_MIDIs, output_MIDI_channels=encode_MIDI_channels, char_offset=chars_encoding_offset, dataset_MIDI_events_time_denominator=time_denominator, output_velocity=encode_velocities, MIDI_channel=desired_MIDI_channel_to_process, transpose_by=-12, MIDI_patch=range(0, 127), flip=True, melody_conditioned_encoding=melody_conditioned_chords, melody_pitch_baseline=melody_pitch_baseline, perfect_timings=perfect_timings, musenet_encoding=MuseNet_encoding, transform=transform_to_pitch)
      TXT_String += TXT
      melody_list_f += melody
      chords_list_f += chords
      INTS_f.append(INTS)
      gfiles += 1

  except KeyboardInterrupt:
    print('Saving current progress and quitting...')
    break  
  
  except:
    print('Bad MIDI:', f)
    continue

TXT_String += 'TOTAL_SONGS_IN_DATASET=' + str(gfiles)

try:
  print('Task complete :)')
  print('==================================================')
  if add_transposed_dataset_by_this_many_pitches != 0:
    print('NOTE: Transposed dataset was added per users request.')
    print('==================================================')
  if add_transposed_and_flipped_dataset == True:
    print('NOTE: Flipped dataset was added per users request.')  
    print('==================================================')
  print('Number of processed dataset MIDI files:', files_count)
  print('Number of MIDI chords recorded:', len(chords_list_f))
  print('First chord event:', chords_list_f[0], 'Last chord event:', chords_list_f[-1]) 
  print('Number of recorded melody events:', len(melody_list_f))
  print('First melody event:', melody_list_f[0], 'Last Melody event:', melody_list_f[-1])
  print('Total number of MIDI events recorded:', len(chords_list_f) + len(melody_list_f))
  print('==================================================')

  # Writing dataset to TXT file
  with open(file_name_to_output_dataset_to + '.txt', 'wb') as f:
    f.write(TXT_String.encode('utf-8', 'replace'))
    f.close

  # Dataset
  MusicDataset = [chords_list_f, melody_list_f, INTS_f]

  # Writing dataset to pickle file
  TMIDIX.Tegridy_Any_Pickle_File_Writer(MusicDataset, file_name_to_output_dataset_to)

except:
  print('=' * 70)
  print('IO Error!')
  print('Please check that Dataset dir is not empty/check other IO code.')
  print('=' * 70)
  print('Shutting down...')
  print('=' * 70)


# In[ ]:


# Process INTs...

INTS_f1 = []


for chords_list in tqdm(chords_list_f):
    INTS_f1.append([-1, -1, -1, -1, -1]) # Intro
    pe = chords_list[0]
    count = 0
    for i in chords_list:

        INTS_f1.append([int(abs(i[1]-pe[1])/ 10), int(i[2] / 10), i[4], i[3], i[5] ])
        
        if count == len(chords_list)-50:
            INTS_f1.append([-2, -2, -2, -2, -2]) # Outro
        
        count += 1
        pe = i
    INTS_f1.append([-3, -3, -3, -3, -3]) # End


# In[ ]:


TMIDIX.Tegridy_Any_Pickle_File_Writer(INTS_f1, '/notebooks/Morpheus_INTS_f1-MI-E')


# In[ ]:


INTS_f1 = TMIDIX.Tegridy_Any_Pickle_File_Reader('/notebooks/Morpheus_INTS_f1-MI-E')


# In[ ]:


#@title Load processed INTs datasets
number_of_batches = 32 #@param {type:"slider", min:2, max:32, step:2}
n_workers = 8
dataset_ratio = 0.5

print('=' * 50)
print('Prepping INTs datasets...')


train_data1 = []

avg_vel = int(sum([y[4] for y in INTS_f1]) / len(INTS_f1))

pe = INTS_f1[0]

for i in tqdm(INTS_f1):
  if max(i) < 256 and min(i) >= 0 and i[3] < 10:

    if i[0] != 0:
        train_data1.extend([i[0] + (int(i[1] / 25) * 256)])

    if i[4] > avg_vel: 
        train_data1.extend([(256 * 11) + 128 + (256 * i[3])+i[2]])
    else:
        train_data1.extend([(256 * 11) + (256 * i[3])+i[2]])
        
    pe = i
  
  if max(i) == -1 and min(i) == -1: # Intro
      train_data1.extend([(256 * 11)+(256 * 11)-3])
  
  if max(i) == -2 and min(i) == -2: # Outro
      train_data1.extend([(256 * 11)+(256 * 11)-2])
  
  if max(i) == -3 and min(i) == -3: # End
      train_data1.extend([(256 * 11)+(256 * 11)-1])

train_data = train_data1[:int(len(train_data1) * dataset_ratio)]

val_dataset = train_data[:int(len(train_data) * 0.03)]
test_dataset = train_data[:int(len(train_data) * 0.03)]

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
for x, tgt in tqdm(train_loader):
    print(f'X shape: {x.shape}')
    print(f'Target shape: {tgt.shape}')
    break
print('=' * 50)

print('Validation loader')
for x, tgt in tqdm(val_loader):
    print(f'X shape: {x.shape}')
    print(f'Target shape: {tgt.shape}')
    break
print('=' * 50)

print('Test loader')
for x, tgt in tqdm(test_loader):
    print(f'X shape: {x.shape}')
    print(f'Target shape: {tgt.shape}')
    break
print('=' * 50)

print('Done! Enjoy! :)')
print('=' * 50)


# # Test the resulting INTs dataset...

# In[ ]:


train_data


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
    if s >= 0 and s <= 256 * 11:
        time += s % 256
        dur = ((s // 256) + 1) * 250
    
    if s >= 256 * 11 and s < (256 * 21):
        if (s // 128) % 2 != 0:
            vel = 90
            channel = ((s-128-(256*11)) // 256)
        else:
            vel = 60
            channel = ((s-(256*11)) // 256)
        
        pitch = s % 256
        
        song_f.append(['note', (abs(time))*10, dur, channel, pitch, vel ])
    
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
config = GPTConfig(5640, 
                   max_seq,
                   dim_feedforward=1024,
                   n_layer=6, 
                   n_head=8, 
                   n_embd=1024,
                   enable_rpr=True,
                   er_len=max_seq)
model = GPT(config).to(get_device())

#=====

init_step = 0
lr = LR_DEFAULT_START
lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)
eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
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
    
    loss = train(epoch+1, model, train_loader, train_loss_func, opt, lr_scheduler, num_iters=-1, save_checkpoint_steps=4000)
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
eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
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
full_path_to_model_checkpoint = "/notebooks/Morpheus-Trained-Model-1024-MI.pth" #@param {type:"string"}
torch.save(model.state_dict(), full_path_to_model_checkpoint)
print('Done!')


# # Congrats! You did it! :)
