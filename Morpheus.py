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


# # (MODEL)

# # (LOAD)

# In[ ]:


#@title Load/Reload the model
full_path_to_model_checkpoint = "/notebooks/Morpheus-Trained-Model.pth" #@param {type:"string"}

print('Loading the model...')
config = GPTConfig(5640, 
                   max_seq,
                   dim_feedforward=1024,
                   n_layer=6, 
                   n_head=8, 
                   n_embd=1024,
                   enable_rpr=True,
                   er_len=max_seq)

model = GPT(config).to(get_device())

model.load_state_dict(torch.load(full_path_to_model_checkpoint))

model.eval()
print('Done!')


# # (GENERATE MUSIC)

# ## Custom MIDI option

# In[ ]:


#@title Load your custom MIDI here
full_path_tp_custom_MIDI = "/notebooks/tegridy-tools/tegridy-tools/seed2.mid" #@param {type:"string"}
print('=' * 70)

print('Loading custom MIDI...')

print('File name:', full_path_tp_custom_MIDI)

data = TMIDIX.Optimus_MIDI_TXT_Processor(full_path_tp_custom_MIDI, 
                                         dataset_MIDI_events_time_denominator=10, 
                                         perfect_timings=True, 
                                         musenet_encoding=True, 
                                         char_offset=0, 
                                         MIDI_channel=16, 
                                         MIDI_patch=range(0, 127)
                                        )
print('=' * 70)
print('Converting to INTs...')

times = []
pitches = []

itimes = []
ipitches = []

melody = []

SONG = data[5]
inputs = []

for i in SONG:
    if max(i) < 256 and min(i) >= 0 and i[2] < 10:

        if i[0] != 0:
            inputs.extend([i[0] + (int(i[1] / 25) * 256)])
            
            melody.extend([i[0] + (int(i[1] / 25) * 256)])
            
            if i[4] > 84:
                melody.extend([(256 * 11) + 128 + (256 * i[2])+i[3]])
            else:
                melody.extend([(256 * 11) + (256 * i[2])+i[3]])

            if i[2] < 10:
              times.extend([i[0] + (int(i[1] / 25) * 256)])
              
              if i[4] > 84:
                  pitches.extend([(256 * 11) + 128 + (256 * i[2])+i[3]])
              else:
                  pitches.extend([(256 * 11) + (256 * i[2])+i[3]])

        if i[4] > 84:
            inputs.extend([(256 * 11) + 128 + (256 * i[2])+i[3]])
        else:
            inputs.extend([(256 * 11) + (256 * i[2])+i[3]])
        
        if i[2] < 10:
              itimes.extend([i[0] + (int(i[1] / 25) * 256)])
              
              if i[4] > 84:
                  ipitches.extend([(256 * 11) + 128 + (256 * i[2])+i[3]])
              else:
                  ipitches.extend([(256 * 11) + (256 * i[2])+i[3]])
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

temperature = 1 #@param {type:"slider", min:0.1, max:1.3, step:0.1}

show_stats = True #@param {type:"boolean"}

number_of_instruments = 1

#===================================================================

tokens_range = (256 * 11) + (256 * number_of_instruments)

fname = '/notebooks/Morpheus-Music-Composition'

print('Morpheus Music Model Continuation Generator')

output_signature = 'Morpheus'
song_name = 'RGA Composition'
out = []
if show_stats:
  print('=' * 70)
  print('Priming type:', priming_type)
  print('Custom MIDI trim type:', custom_MIDI_trim_type)
  print('Temperature:', temperature)
  print('Tokens range:', tokens_range)

print('=' * 70)
if priming_type == 'Intro':
    rand_seq = model.generate(torch.Tensor([(256 * 11)+(256 * 11)-1, 
                                            (256 * 11)+(256 * 11)-3]), 
                                            target_seq_length=number_of_tokens_to_generate,
                                            temperature=temperature,
                                            stop_token=tokens_range,
                                            verbose=show_stats)
    
    out = rand_seq[0].cpu().numpy().tolist()

if priming_type == 'Outro':
    rand_seq = model.generate(torch.Tensor([(256 * 11)+(256 * 11)-2]), 
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
    dur = 0
    vel = 0
    pitch = 0
    duration = 0
    once = True
    for s in song:
        if s >= 0 and s <= 256 * 11:
            time += s % 256
            dur = ((s // 256) + 1) * 250

        if s >= 256 * 11 and s < (256 * 21):
            if (s // 128) % 2 != 0:
                vel = 80 + (s % 256) % 24
                channel = ((s-128-(256*11)) // 256)
            else:
                vel = 64 + (s % 256) % 24
                channel = ((s-(256*11)) // 256)

            pitch = s % 256

            song_f.append(['note', (abs(time))*10, dur, channel, pitch, vel ])

            if song.index(s) >= len(sequence) and once:
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
output_original_and_generated_melodies = False
print('=' * 70)


print('Morpheus Music Model Melody Generator')
print('=' * 70)

song = []
sng = melody[:number_of_input_melody_notes * 2]

for i in tqdm(range(number_of_input_melody_notes)):
  
  if len(sng)+2  >= 1024:
    break
  

  rand_seq = model.generate(torch.Tensor(sng), 
                              target_seq_length=len(sng) + 2,
                              temperature=1,
                              stop_token=(256*11)+(256 * number_of_instruments),
                              verbose=False)

  out = rand_seq[0].cpu().numpy().tolist()

  if out[-2] < 256 * 11 and out[-1] > 256 * 11:
      sng.extend(out[-2:])

print('=' * 70)
print('Converting to MIDI...')

if len(sng) != 0:
    song = []
    if output_original_and_generated_melodies:
        song = sng
    else:
        song = sng[len(melody[:number_of_input_melody_notes * 2]):]
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    duration = 0
    once = True
    for s in song:
        if s >= 0 and s <= 256 * 11:
            time += s % 256
            dur = ((s // 256) + 1) * 250

        if s >= 256 * 11 and s < (256 * 21):
            if (s // 128) % 2 != 0:
                vel = 80 + (s % 256) % 24
                channel = ((s-128-(256*11)) // 256)
            else:
                vel = 64 + (s % 256) % 24
                channel = ((s-(256*11)) // 256)

            pitch = s % 256

            song_f.append(['note', (abs(time))*10, dur, channel, pitch, vel ])
      
    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Morpheus',  
                                                          output_file_name = '/notebooks/Morpheus-Music-Composition', 
                                                          track_name='Project Los Angeles', 
                                                          number_of_ticks_per_quarter=500)

    print('Done!')

print('=' * 70)

times = []
pitches = []

if output_original_and_generated_melodies:
    for i in range(0, len(sng)-2, 2):
       times.append(sng[i])
       pitches.append(sng[i+1])

else:
    for i in range(len(melody[:number_of_input_melody_notes * 2]), len(sng)-2, 2):
       times.append(sng[i])
       pitches.append(sng[i+1])


# # Accompaniment Generation

# ## Simple Accompaniment Generator

# In[ ]:


#@title Generate an accompaniment for the custom MIDI melody
number_of_input_melody_notes = 2560 #@param {type:"slider", min:16, max:256, step:16}
number_of_instruments = 10
number_of_prime_notes = 0

print('=' * 70)


print('Morpheus Music Model Accompaniment Generator')
print('=' * 70)

song = []
sng = []

for i in range(number_of_prime_notes):
    sng.append(times[i])
    sng.append(pitches[i])
    
for i in tqdm(range(number_of_prime_notes, min(number_of_input_melody_notes, len(pitches)))):
  
  #if len(sng + [times[i], pitches[i]]) + 16 >= 1024:
    #break
  
  rand_seq = model.generate(torch.Tensor(sng[-1006:] + [times[i], pitches[i]]), 
                              target_seq_length=len(sng[-1006:]) + 2 + 16, 
                              temperature=1,
                              stop_token=(256*11)+(256 * number_of_instruments),
                              verbose=False)
    
  out = rand_seq[0].cpu().numpy().tolist()

  outy = []
  
  for o in out[len(sng[-1006:])+2:]:
    if o >=256 * 11:
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
    dur = 0
    vel = 0
    pitch = 0
    duration = 0
    once = True
    for s in song:
        if s >= 0 and s <= 256 * 11:
            time += s % 256
            dur = ((s // 256) + 1) * 250

        if s >= 256 * 11 and s < (256 * 21):
            if (s // 128) % 2 != 0:
                vel = 80 + (s % 256) % 24
                channel = ((s-128-(256*11)) // 256)
            else:
                vel = 64 + (s % 256) % 24
                channel = ((s-(256*11)) // 256)

            pitch = s % 256

            song_f.append(['note', (abs(time))*10, dur, channel, pitch, vel ])
      
    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Morpheus',  
                                                          output_file_name = '/notebooks/Morpheus-Music-Composition', 
                                                          track_name='Project Los Angeles', 
                                                          number_of_ticks_per_quarter=500)

    print('Done!')

print('=' * 70)


# ## Advanced Accompaniment Generator

# In[ ]:


#@title Generate an accompaniment for the custom MIDI melody
number_of_input_melody_notes = 2560 #@param {type:"slider", min:16, max:256, step:16}
number_of_instruments = 10
minimum_beat_delta_time = 12
number_of_prime_notes = 0

print('=' * 70)


print('Morpheus Music Model Advanced Accompaniment Generator')
print('=' * 70)

song = []
sng = []
tim = 0

for i in range(number_of_prime_notes):
    sng.append(times[i])
    sng.append(pitches[i])

for i in tqdm(range(number_of_prime_notes, min(number_of_input_melody_notes, len(pitches))-1)):
  
  #if len(sng) + 2 + 16 >= 1024:
    #break
  
  rand_seq = model.generate(torch.Tensor(sng[-1006:] + [abs(times[i]-tim), pitches[i]]), 
                              target_seq_length=len(sng[-1006:]) + 2 + 16, 
                              stop_token=(256*11)+(256 * number_of_instruments),
                              verbose=False)
    
  out = rand_seq[0].cpu().numpy().tolist()
    
  sng.extend([abs(times[i]-tim), pitches[i]])

  outy = []
  tim = 0

  for o in out[len(sng[-1006:])+2:]:
    if o >=(256*11):
      outy.append(o)

    else:
      if ((times[i+1] % 256) -tim)  > o % 256 and o % 256 > minimum_beat_delta_time * 2:
         outy.append(o)
         tim += o % 256 
      else:
         break

  sng.extend(outy)

print('=' * 70)
print('Converting to MIDI...')

if len(sng) != 0:
    song = []
    song = sng
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    duration = 0
    once = True
    for s in song:
        if s >= 0 and s <= 256 * 11:
            time += s % 256
            dur = ((s // 256) + 1) * 250

        if s >= 256 * 11 and s < (256 * 21):
            if (s // 128) % 2 != 0:
                vel = 80 + (s % 256) % 24
                channel = ((s-128-(256*11)) // 256)
            
            else:
                vel = 64 + (s % 256) % 24
                channel = ((s-(256*11)) // 256)

            pitch = s % 256

            song_f.append(['note', (abs(time))*10, dur, channel, pitch, vel ])
          
      
    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Morpheus',  
                                                          output_file_name = '/notebooks/Morpheus-Music-Composition', 
                                                          track_name='Project Los Angeles', 
                                                          number_of_ticks_per_quarter=500)

else:
  print('Models output is empty! Check the code...')
  print('Shutting down...')

print('=' * 70)


# # Pitches Inpainting

# In[ ]:


#@title Generate an accompaniment for the custom MIDI melody
number_of_input_melody_notes = 400 #@param {type:"slider", min:16, max:256, step:16}
number_of_instruments = 3
number_of_prime_notes = 64
original_pitch_ratio = 4

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
                                  stop_token=(256*11)+(256 * number_of_instruments),
                                  verbose=False)

      out = rand_seq[0].cpu().numpy().tolist()
        
      if out[-1] > 256 * 11:
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
    dur = 0
    vel = 0
    pitch = 0
    duration = 0
    once = True
    for s in song:
        if s >= 0 and s <= 256 * 11:
            time += s % 256
            dur = ((s // 256) + 1) * 250

        if s >= 256 * 11 and s < (256 * 21):
            if (s // 128) % 2 != 0:
                vel = 80 + (s % 256) % 24
                channel = ((s-128-(256*11)) // 256)
            
            else:
                vel = 64 + (s % 256) % 24
                channel = ((s-(256*11)) // 256)

            pitch = s % 256

            song_f.append(['note', (abs(time))*10, dur, channel, pitch, vel ])
          
      
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
