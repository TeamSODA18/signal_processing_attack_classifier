# signal_processing_attack_classifier

Package for inferening signal processing attacked audio files

## Leaderboard

| Attacked Model | Dataset       | AUC      | Accuracy | Precision | Recall   | F1 Score |
|----------------|---------------|----------|----------|-----------|----------|----------|
| whisper        | librispeech   | 0.9886   | 0.9667   | 0.9879    | 0.9455   | 0.9663   |
| assembly       | librispeech   | 0.9214   | 0.8133   | 0.7798    | 0.8733   | 0.8239   |
| whisper        | commonvoice   | 0.8883   | 0.7633   | 0.7159    | 0.8733   | 0.7868   |
| assembly       | commonvoice   | 0.8289   | 0.73     | 0.6994    | 0.8067   | 0.7492   |

## Installation

```
pip install git+https://github.com/TeamSODA18/signal_processing_attack_classifier.git
```

## Usage

### Mel spectogram features using audio array
```
from spClassify.infer import SpMelCl
import torchaudio

classifier = SpMelCl()

data, sample_rate = torchaudio.load(r'path\to\audio\file')

print(classifier(data, sample_rate))
```

### Using audio path
```
from spClassify.infer import SpAttackCl
from spClassify.models import MLP

classifier = SpAttackCl()

print(classifier(r'path\to\audio\file'))
```

### Using audio array
```
from spClassify.infer import SpAttackClTensor
from spClassify.models import MLP
import torchaudio

classifier = SpAttackClTensor()

data, sample_rate = torchaudio.load(r'path\to\audio\file')

print(classifier(data, sample_rate))
```

### Whisper features using audio array
```
from spClassify.infer import WhisperClTensor
from spClassify.models import WhisperCl
import torchaudio

classifier = WhisperClTensor()

data, sample_rate = torchaudio.load(r'path\to\audio\file')

print(classifier(data, sample_rate))
```