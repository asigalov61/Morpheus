# Morpheus Contrastive Language-Music Pretraining

***

## DEMO CLMP MODEL

### This is the model that can generate music based on the language (think CLIP for Music).

### This is a DEMO model, so it does not always produces coherent melody but it does play quite well

### Trained on LAKH/clean_midi MIDI subset (~ 9000 excerpts and composition names)

***

### Download here:

```
!wget --no-check-certificate -O 'Morpheus-CLMP-Trained-Model.pth' "https://onedrive.live.com/download?cid=8A0D502FC99C608F&resid=8A0D502FC99C608F%2118523&authkey=AKWZMkzvZv3WBSo"
```

### How to use:

#### LOAD:

```
#@title Load/Reload the model
full_path_to_model_checkpoint = "./Morpheus-CLMP-Trained-Model.pth" #@param {type:"string"}

print('Loading the model...')
config = GPTConfig(5890, 
                   1024,
                   dim_feedforward=1024,
                   n_layer=6, 
                   n_head=8, 
                   n_embd=1024,
                   enable_rpr=True,
                   er_len=1024)

model = GPT(config).to(get_device())

model.load_state_dict(torch.load(full_path_to_model_checkpoint))

model.eval()
print('Done!')
```

#### INPUT SEQ:

```
def str2ints(string):
    return [5888] + [ord(y)+(256 * 11)+(256 * 11) for y in string] + [5888]

sequence = str2ints('love me tonight')
```

#### GENERATION:

```
number_of_instruments = 12
number_of_tokens_to_generate = 1024 #@param {type:"slider", min:512, max:1024, step:8}

temperature = 0.8 #@param {type:"slider", min:0.1, max:1.3, step:0.1}

show_stats = True #@param {type:"boolean"}

#===================================================================

tokens_range = (256 * 11) + (256 * number_of_instruments)

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
                                                          output_file_name = './Morpheus-Music-Composition', 
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
```

#### LANGUAGE OUTPUT SEQUENCE (sometimes the model wants to speak-up, so you can decode the language with the following code)

```
def ints2string(ints):
    return ''.join([chr(y-(256 * 11)-(256 * 11)) for y in ints])

ints2string(out)
```

***

### Project Los Angeles

### Tegridy Code 2022
