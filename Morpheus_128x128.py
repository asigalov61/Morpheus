#!/usr/bin/env python
# coding: utf-8

# # Morpheus (ver. 1.0)
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
import copy
import tqdm as tqdm


if not os.path.exists('/notebooks/Dataset'):
    os.makedirs('/notebooks/Dataset')

print('Loading TMIDIX module...')
os.chdir('/notebooks/tegridy-tools/tegridy-tools')
import TMIDIX

os.chdir('/notebooks/tegridy-tools/tegridy-tools')
from GPT2RGAX import *

import matplotlib.pyplot as plt

os.chdir('/notebooks/')


# # (MODEL)

# # (LOAD)

# In[ ]:


#@title Load/Reload the model

from collections import OrderedDict

full_path_to_model_checkpoint = "/notebooks/Morpheus-Trained-Model-2048.pth" #@param {type:"string"}

print('Loading the model...')
config = GPTConfig(19200, 
                   max_seq,
                   dim_feedforward=1024,
                   n_layer=16, 
                   n_head=16, 
                   n_embd=1024,
                   enable_rpr=True,
                   er_len=max_seq)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(config)

state_dict = torch.load(full_path_to_model_checkpoint, map_location=device)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
   name = k[7:] #remove 'module'
   new_state_dict[name] = v

model.load_state_dict(new_state_dict)

model.to(device)

model.eval()

print('Done!')


# # (GENERATE MUSIC)

# ## Custom MIDI option

# In[ ]:


f = '/notebooks/tegridy-tools/tegridy-tools/seed2.mid'
SONG = []
#print('Loading MIDI file...')
score = TMIDIX.midi2ms_score(open(f, 'rb').read())

events_matrix = []

itrack = 1
stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

        time = min(127, int(abs(e[1]-pe[1]) / 10))
        dur = min(127, int(e[2] / 10))
        cha = e[3]
        ptc = e[4]
        vel = e[5]

        SONG.append([time, dur, ptc, cha, vel])

        pe = e
        
        
#====================================

print('=' * 70)
print('Converting to INTs...')

times = []
pitches = []

itimes = []
ipitches = []

melody = []

inputs = []

for i in SONG:
    if max(i) < 128 and min(i) >= 0:

        #if i[0] != 0:
        inputs.extend([i[0] + int(i[1] * 128)])

        melody.extend([i[0] + int(i[1] * 128)])

        if i[4] > 84:
            melody.extend([(128*128) + 128 + (256 * i[3])+i[2]])
        else:
            melody.extend([(128*128) + (256 * i[3])+i[2]])

        if i[3] < 10:
          times.extend([i[0] + int(i[1] * 128)])

          if i[4] > 84:
              pitches.extend([(128*128) + 128 + (256 * i[3])+i[2]])
          else:
              pitches.extend([(128*128) + (256 * i[3])+i[2]])

        if i[4] > 84:
            inputs.extend([(128*128) + 128 + (256 * i[3])+i[2]])
        else:
            inputs.extend([(128*128) + (256 * i[3])+i[2]])
        
        if i[3] < 10:
              itimes.extend([i[0] + int(i[1] * 128)])
              
              if i[4] > 84:
                  ipitches.extend([(128*128) + 128 + (256 * i[3])+i[2]])
              else:
                  ipitches.extend([(128*128) + (256 * i[3])+i[2]])
        pe = i

print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)


# # Continuation Generation

# In[ ]:


#@title Generate and download a MIDI file

#@markdown NOTE: The first continuation sample may not be perfect, so generate several samples if you are not getting good results

number_of_tokens_to_generate = 1024 #@param {type:"slider", min:512, max:1024, step:8}
priming_type = "Custom MIDI" #@param ["Intro", "Outro", "Custom MIDI"]
custom_MIDI_trim_type = "From Start" #@param ["From Start", "From End"]

temperature = 0.8 #@param {type:"slider", min:0.1, max:1.3, step:0.1}

show_stats = True #@param {type:"boolean"}

number_of_instruments = 1

#===================================================================

tokens_range = (128*128) + (256 * number_of_instruments)

fname = '/notebooks/Morpheus-Music-Composition'

print('Morpheus Music Model Continuation Generator')

output_signature = 'Morpheus'
song_name = 'RGA Composition'
out = []
sequence = []
if show_stats:
  print('=' * 70)
  print('Priming type:', priming_type)
  print('Custom MIDI trim type:', custom_MIDI_trim_type)
  print('Temperature:', temperature)
  print('Tokens range:', tokens_range)

print('=' * 70)
if priming_type == 'Intro':
    rand_seq = model.generate(torch.Tensor([(128*128)+(256 * 11)-1, 
                                            (128*128)+(256 * 11)-3]), 
                                            target_seq_length=number_of_tokens_to_generate,
                                            temperature=temperature,
                                            stop_token=tokens_range,
                                            verbose=show_stats)
    
    out = rand_seq[0].cpu().numpy().tolist()

if priming_type == 'Outro':
    rand_seq = model.generate(torch.Tensor([(128*128)+(256 * 11)-2]), 
                              target_seq_length=number_of_tokens_to_generate,
                              temperature=temperature,
                              stop_token=tokens_range,
                              verbose=show_stats)
    
    out = rand_seq[0].cpu().numpy().tolist()

if priming_type == 'Custom MIDI' and inputs != []:
    out = []

    if custom_MIDI_trim_type == 'From Start':
      sequence = inputs[:512]
    else:
      sequence = inputs[-512:]

    rand_seq = model.generate(torch.Tensor(sequence), 
                              target_seq_length=number_of_tokens_to_generate, 
                              temperature=temperature,
                              stop_token=tokens_range,
                              verbose=show_stats)
    
    out = rand_seq[0].cpu().numpy().tolist()

print('=' * 70)

if len(out) != 0:
    song = []
    song = out
    song_f = []
    time = 0
    dur = 1
    vel = 0
    pitch = 0
    once = 0
    duration = 0
    for s in song:
        if s >= 0 and s < 128 * 128:
            time += (s % 128) * 10
            dur = (s // 128) * 10

        if s >= 128 * 128 and s < (128 * 128) + (256 * 11):
            if (s // 128) % 2 != 0:
                vel = 90
                channel = ((s-128-(128 * 128)) // 256)
            else:
                vel = 60
                channel = ((s-(128 * 128)) // 256)

            pitch = s % 256

            song_f.append(['note', abs(time), dur, channel, pitch, vel ])


            if len(song_f) >= len(sequence) and once:
                song_f.append(['text_event', abs(time) * 10, 'Continuation Start Here'])
                once = False

    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Morpheus',  
                                                          output_file_name = '/notebooks/Morpheus-Music-Composition', 
                                                          track_name='Project Los Angeles', 
                                                          number_of_ticks_per_quarter=500)

    print('Done!')

    if show_stats:
      print('=' * 70)
      print('Detailed MIDI stats:')
      for key, value in detailed_stats.items():
            print('=' * 70)
            print(key, '|', value)

    print('=' * 70)

else:
  print('Models output is empty! Check the code...')
  print('Shutting down...')

print('=' * 70)


# # Melody Generator

# In[ ]:


#@title Generate an accompaniment for the custom MIDI melody
number_of_input_melody_notes = 128 #@param {type:"slider", min:16, max:256, step:16}
number_of_instruments = 1
temperature = 1

print('=' * 70)


print('Morpheus Music Model Melody Generator')
print('=' * 70)

song = []
sng = copy.deepcopy(melody[:number_of_input_melody_notes])

for i in tqdm(range(number_of_input_melody_notes)):
  
  if len(sng)+2  >= 1024:
    break
  

  rand_seq = model.generate(torch.Tensor(sng), 
                              target_seq_length=len(sng) + 2,
                              temperature=temperature,
                              stop_token=(128*128)+(256 * number_of_instruments),
                              verbose=False)

  out = rand_seq[0].cpu().numpy().tolist()

  #if out[-2] < 128 * 128 and out[-1] > 128 * 128:
  sng.extend(out[-2:])

print('=' * 70)
print('Converting to MIDI...')

if len(sng) != 0:
    song = []
    
    song = sng
   
    song = sng[len(melody[:number_of_input_melody_notes * 2]):]
    song_f = []
    time = 0
    dur = 1
    vel = 0
    pitch = 0
    duration = 0
    for s in song:
        if s >= 0 and s < 128 * 128:
            time += (s % 128) * 10
            dur = (s // 128) * 10

        if s >= 128 * 128 and s < (128 * 128) + (256 * 11):
            if (s // 128) % 2 != 0:
                vel = 90
                channel = ((s-128-(128 * 128)) // 256)
            else:
                vel = 60
                channel = ((s-(128 * 128)) // 256)

            pitch = s % 256

            song_f.append(['note', abs(time), dur, channel, pitch, vel ])
      
    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Morpheus',  
                                                          output_file_name = '/notebooks/Morpheus-Music-Composition', 
                                                          track_name='Project Los Angeles', 
                                                          number_of_ticks_per_quarter=500)

    print('Done!')

print('=' * 70)


# # Accompaniment Generation

# ## Simple Accompaniment Generator

# In[ ]:


#@title Generate an accompaniment for the custom MIDI melody
number_of_input_melody_notes = 256 #@param {type:"slider", min:16, max:256, step:16}
number_of_instruments = 10
number_of_prime_notes = 0
temperature = 0.8
print('=' * 70)


print('Morpheus Music Model Accompaniment Generator')
print('=' * 70)

song = []
sng = []

for i in range(number_of_prime_notes):
    sng.append(times[i])
    sng.append(pitches[i])
    
for i in tqdm(range(number_of_prime_notes, min(number_of_input_melody_notes, len(pitches)))):
  
    if len(sng) + 16 >= 1024:
        break

    rand_seq = model.generate(torch.Tensor(sng[-1006:] + [times[i], pitches[i]]), 
                              target_seq_length=len(sng[-1006:]) + 2 + 16, 
                              temperature=temperature,
                              stop_token=(128*128)+(256 * number_of_instruments),
                              verbose=False)

    out = rand_seq[0].cpu().numpy().tolist()

    outy = []

    for o in out[len(sng[-1006:])+2:]:
        if o < 128*128:
            time = o % 128
        
        
        
        if time == 0:
            outy.append(o)
        else:
            break
    sng.extend([times[i], pitches[i]])
    sng.extend(outy)
    # print(len(outy))

print('=' * 70)
print('Converting to MIDI...')

if len(sng) != 0:
    song = []
    song = sng
    song_f = []
    time = 0
    dur = 1
    vel = 0
    pitch = 0
    duration = 0
    for s in song:
        if s >= 0 and s < 128 * 128:
            time += (s % 128) * 10
            dur = (s // 128) * 10

        if s >= 128 * 128 and s < (128 * 128) + (256 * 11):
            if (s // 128) % 2 != 0:
                vel = 90
                channel = ((s-128-(128 * 128)) // 256)
            else:
                vel = 60
                channel = ((s-(128 * 128)) // 256)

            pitch = s % 256

            song_f.append(['note', abs(time), dur, channel, pitch, vel ])
            
    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Morpheus',  
                                                          output_file_name = '/notebooks/Morpheus-Music-Composition', 
                                                          track_name='Project Los Angeles', 
                                                          number_of_ticks_per_quarter=500)

    print('Done!')

print('=' * 70) 


# # Pitches Inpainting

# In[ ]:


#@title Generate an accompaniment for the custom MIDI melody
number_of_input_melody_notes = 512 #@param {type:"slider", min:16, max:256, step:16}
number_of_instruments = 1
number_of_prime_notes = 32
original_pitch_ratio = 2

print('=' * 70)


print('Morpheus Music Model Pitches Inpainting Generator')
print('=' * 70)

song = []
sng = []
tim = 0
out = [0]

for i in range(number_of_prime_notes):
    sng.append(itimes[i])
    sng.append(ipitches[i])

for i in tqdm(range(number_of_prime_notes, min(number_of_input_melody_notes, len(ipitches))-1)):
  
  if len(sng) + 2 >= 1024:
    break
  
  for j in range(100):

      rand_seq = model.generate(torch.Tensor(sng + [abs(itimes[i]) ]), 
                                  target_seq_length=len(sng) + 2, 
                                  stop_token=(128*128)+(256 * number_of_instruments),
                                  verbose=False)

      out = rand_seq[0].cpu().numpy().tolist()
        
      if out[-1] > 128 * 128:
        break
    
  sng.extend([abs(itimes[i])])
  
  if i % original_pitch_ratio == 0:
    sng.extend([pitches[i]])
  
  else:
    sng.extend([out[-1]])

print('=' * 70)
print('Converting to MIDI...')

if len(sng) != 0:
    song = []
    song = sng
    song_f = []
    time = 0
    dur = 1
    vel = 0
    pitch = 0
    duration = 0
    for s in song:
        if s >= 0 and s < 128 * 128:
            time += (s % 128) * 10
            dur = (s // 128) * 10

        if s >= 128 * 128 and s < (128 * 128) + (256 * 11):
            if (s // 128) % 2 != 0:
                vel = 90
                channel = ((s-128-(128 * 128)) // 256)
            else:
                vel = 60
                channel = ((s-(128 * 128)) // 256)

            pitch = s % 256

            song_f.append(['note', abs(time), dur, channel, pitch, vel ])
          
      
    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Morpheus',  
                                                          output_file_name = '/notebooks/Morpheus-Music-Composition', 
                                                          track_name='Project Los Angeles', 
                                                          number_of_ticks_per_quarter=500)

else:
  print('Models output is empty! Check the code...')
  print('Shutting down...')

print('=' * 70)


# # Congrats! You did it! :)
