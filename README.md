# signal_processing_attack_classifier

Package for inferening signal processing attacked audio files

## Installation

```
pip install git+https://github.com/TeamSODA18/signal_processing_attack_classifier.git
```

## Usage

```
from spClassify.infer import SpAttackCl
from spClassify.models import MLP

classifier = SpAttackCl()

print(classifier(r'path\to\audio\file'))
```