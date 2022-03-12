# Morpheus Training Code and Info

***

### Trained with partial [LAKH MuseNet Dataset](https://github.com/asigalov61/LAKH-MuseNet-MIDI-Dataset)

### Please see Morpheus Maker colab notebook for the original training code

### Enjoy!

***

### Version 2.0 NOTE:

#### Morpheus Maker 2.0 uses Torch's nn.DataParallel wrapper which requires special loading procedure to work. You will need to use the following code to load trained models:

```
#@title Load/Reload the model

from collections import OrderedDict

full_path_to_model_checkpoint = "/notebooks/gpt2_rpr_checkpoint_1_epoch_24000_steps_0.0374_loss.pth" #@param {type:"string"}

print('Loading the model...')
config = GPTConfig(5640, 
                   max_seq,
                   dim_feedforward=1024,
                   n_layer=8, 
                   n_head=8, 
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
```

***

### Project Los Angeles

### Tegridy Code 2021
