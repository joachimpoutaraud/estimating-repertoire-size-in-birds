# UNSUPERVISED BIRD SONG SYLLABLES CLASSIFICATION

![chloris](https://cdn.download.ams.birds.cornell.edu/api/v1/asset/44588041/1800)
<h4 align="center">Photo credits © Rogério Rodrigues</h4>



This repository presents an unsupervised method to estimate the size of the repertoire of the [European Greenfinch](https://en.wikipedia.org/wiki/European_greenfinch) (Chloris Chloris). The proposed system receives as input a set of audio time series which is segmented and converted into a reduced representation set called a feature vector. The system is finally evaluated using clustering performance metrics to find the ideal number of syllables in the data set.

## Installation

Prepare your environment

```python
conda create --name chloris
conda activate chloris
```
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraires

```python
conda install -c anaconda pip
pip install -r requirements.txt
``` 
## Usage
### 1. Downloading

The data set contains a total of 339 audio recordings downloaded from the [Xeno-Canto](https://xeno-canto.org/) sound library, a collaborative project dedicated to sharing bird sounds from all over the world. We use an [area-based query](https://xeno-canto.org/api/2/recordings?query=chloris+area:europe+q:a) gathering European recordings of the [European Greenfinch](https://en.wikipedia.org/wiki/European_greenfinch) (Chloris Chloris). We select only high quality recordings according to the Xeno-Canto quality ratings ranging from **A** (highest quality) to **E** (lowest quality) and remove recordings that have an other species referenced in the background.

To download the data set, you need to install [`wget`](https://www.gnu.org/software/wget/)

- For MAC OS, write the following command in the terminal: `brew install wget`
- For Windows, download the package, copy the `wget.exe` file into your `C:\Windows\System32` folder and run `wget` on the command line to see if it is correctly installed.

Finally, you can run the following script

```
python _downloading.py
```

### 2. Segmenting

Segmentation is a preliminary phase for the analysis and classification of bird syllables. This makes it easier to build analysis and classification systems with segmented objects than with raw data and reduces the size of the dataset which will facilitate computer processing and make it easier to carry on recognition and retrieval. 

Here, input signals are transformed using the [Continuous Wavelet Transform](https://en.wikipedia.org/wiki/Continuous_wavelet_transform#:~:text=In%20mathematics%2C%20the%20continuous%20wavelet,of%20the%20wavelets%20vary%20continuously.) (CWT). Transformation process is computed with the free library for the Python programming language [PyWavelets](https://pypi.org/project/PyWavelets). Aditionally, we calculate the energy envelope of each wavelet vector using Root Mean Square (RMS) energy function. That way, we can isolate segments by finding high-energy peaks in the energy envelope and apply a threshold mask set to -20 dB.

To segment audio recordings, you can run the following script

```
python _segmenting.py
```

### 3. Extracting

Audio features are extracted from bird syllables which are then used as patterns. Classification is done based on the time and frequency domain. The features are calculated from the segmented syllables in order to be classified or recognized. Features constitute a feature vector, which is a representation of the syllable.

To extract the time and frequency domain features, you can run the following script

```
python _extracting.py
```
>**Note:** the features have already been extracted and stored in the `dataset.csv` file 

### 4. Evaluating

![chloris](https://github.com/joachimpoutaraud/estimating-repertoire-size-in-a-songbird/blob/master/notebooks/images/dbscan.jpg)

After features are extracted from augmented data, they are normalized using the max–min method, selected based on individual ranking, and fed into an unsupervised algorithm to automatically cluster bird syllables in the audio recordings.

To evaluate the proposed system, you can run the following script

```
python _evaluating.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.
