import torch
from torchaudio import transforms
import torchaudio
import argparse
import requests

class SpAttackCl:
  def __init__(self):
    print("label 1: attacked")
    print("label 0: original")
    url = 'https://drive.google.com/uc?id=14Gq-X4yhRFYq17hsUfC-WecVqPiMPIsc&export=download'
    r = requests.get(url, allow_redirects=True)
    open('sp-attack-cl.h5', 'wb').write(r.content)
    self.model = torch.load('sp-attack-cl.h5', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    self.mfcc = transforms.MFCC(sample_rate=8000, n_mfcc=256, norm = 'ortho',melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 256, "center": False})
    
  def __call__(self, dir: str):
    data, sample_rate = torchaudio.load(dir)
    resample = transforms.Resample(sample_rate, 8000)
    data = resample(data)
    data = self.mfcc(data)
    data = torch.mean(data,2)
    logits = self.model(data)
    
    confidence = None
    label = None
    if logits[0] >= 6.886499881744385:
      label = 0
      confidence = torch.sigmoid(logits[0])
    else:
      label = 1
      confidence = 1 - torch.sigmoid(logits[0])
    return {'confidence':confidence.detach().item(), 'label':label}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get the label of the audio")
    parser.add_argument('-p', '--path', type=str, action='store', help='path to the audio file')
    args = parser.parse_args()
    
    classifier = SpAttackCl()
    print(classifier(args.path))
