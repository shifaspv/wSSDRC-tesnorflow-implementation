# wSSDRC: a WaveNet based intelligibility modification for improving listening comfort in noise
This is a Tensorflow implementation of the ```wSSDRC``` architecture suggested in <a href="https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2119.pdf"> this paper</a>, where we have suggested a neural speech enrichment approach for intelligibility improvement in noise. The model was trained to generate the same intelligibility gain as the signal processing SSDRC model suggested <a href="https://www.isca-speech.org/archive/archive_papers/interspeech_2012/i12_0635.pdf">in here</a>.

Few samples from the trained model are displayed <a href="https://www.csd.uoc.gr/~shifaspv/IS2018-demo">here</a>.


## Implemented On
Python - 3.6.8 <br>
Tensorflow - 1.14.0 <br>

We required few more very common Python packages, check the ```required.txt``` file and install if you don't have.
## Data set
The model displayed on the paper was trained on the manually created data set with clean speech from <a href="https://datashare.is.ed.ac.uk/handle/10283/1942">here</a>. The model training was done with the intelligibility improved samples from SSDRC model as target, since don't have the right to publish that code, you may please use your own target samples.

Then, generate the lists of wave files ID for training and tessting using the ```./data/generate_wave_id_list.py```, and confirm that the names match to the ones in ```./config/config_params.json```

## Description of the ```./config/config_params.json``` file variables
<table>
  <tr>
    <th>"Name"</th>
    <th>"Discription"</th>
  </tr>
  
  <tr>
    <th>train_id_list:</th>
      <td>list of training wave files ID</td>
  </tr>
    <tr>
    <th>valid_id_list:</th>
      <td>list of validation files ID</td>
  </tr>
  <tr>
    <th>n_channels</th>
    <td>number of channels in each layer</td>
  </tr>
<tr>
    <th>dilations</th>
    <td>dilation rate starting from the begining layer</td>
  </tr>
  <tr>
    <th>target_length</th>
      <td> total samples generated in a single forward epoch</td>
  </tr>
    <tr>
    <th>filter_length</th>
    <td>convolutiona filter width: 3 for non-causal architecture </td>
  </tr>
  <tr>
    <th>Regain</th>
      <td>The level to which wave files are RMS normalised </td>
  </tr>
</table>

The **train_id_list** and **valid_id_list** are generated by ```./data/generate_wave_id_list.py``` file.
## Training the model

Go to the ```./src``` folder and run the ```train.py``` or copy the command below to command line 

```
python train.py
```

Optionally, you can resume the training that could not have been completed, by passing the second argument ```model_id```

```
python train.py --model_id=saved_model_id
```

Trained models will be saved to the ```./saved_models``` directory

## Testing the model

You can use the trained model in ```./saved_model``` directory, or your own model, if you have managed to train the model.
Go to the ```./src``` folder, and compile the ```generate.sh``` file with first argument as the ```model_id```. 

```
./generate.sh saved_model_id
```

A new folder named ```./outputs/saved_model_id``` will be created and saved the output sample.
User can manually edit the wave file ID inside the ```generate.sh```, to generate over multiple files.



