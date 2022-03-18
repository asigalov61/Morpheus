# Morpheus Contrastive Language-Music Pretraining

***

## DEMO CLMP MODEL

### This is the model that can generate music based on the language (think CLIP for Music).

### This is a DEMO model, so it does not always produces coherent melody but it does play quite well

### Trained on LAKH/clean_midi MIDI subset (~ 9000 excerpts and composition names). 
### [Download the datasets here.](https://github.com/asigalov61/Tegridy-MIDI-Dataset/tree/master/CLMP)

***

### Download here:

```
!wget --no-check-certificate -O 'Morpheus-CLMP-Trained-Model.pth' "https://onedrive.live.com/download?cid=8A0D502FC99C608F&resid=8A0D502FC99C608F%2118523&authkey=AKWZMkzvZv3WBSo"
```

### How to use:

#### Please see colab in this section of the repo

### NOTE:
#### LANGUAGE OUTPUT SEQUENCE (sometimes the model wants to speak-up, so you can decode the language with the following code)

```
def ints2string(ints):
    return ''.join([chr(y-(256 * 11)-(256 * 11)) for y in ints])

ints2string(out)
```

***

### Project Los Angeles

### Tegridy Code 2022
