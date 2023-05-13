# signal_processing_attack_classifier

Package for inferening signal processing attacked audio files

## Installation

```
pip install git+https://github.com/TeamSODA18/signal_processing_attack_classifier.git
```

## Usage

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
from spClassify.infer import SpAttackClTensor
from spClassify.models import WhisperCl
import torchaudio

classifier = WhisperClTensor()

data, sample_rate = torchaudio.load(r'path\to\audio\file')

print(classifier(data, sample_rate))
```