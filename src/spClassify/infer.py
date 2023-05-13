import torch
from torchaudio import transforms
import torchaudio
import argparse
import requests
from transformers import WhisperModel, AutoFeatureExtractor


class SpAttackCl:
    def __init__(self):
        print("label 1: attacked")
        print("label 0: original")
        url = "https://drive.google.com/uc?id=14Gq-X4yhRFYq17hsUfC-WecVqPiMPIsc&export=download"
        r = requests.get(url, allow_redirects=True)
        open("sp-attack-cl.h5", "wb").write(r.content)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(
            "sp-attack-cl.h5",
            map_location=self.device,
        )
        self.mfcc = transforms.MFCC(
            sample_rate=8000,
            n_mfcc=256,
            norm="ortho",
            melkwargs={
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 256,
                "center": False,
            },
        )

    def __call__(self, dir: str):
        data, sample_rate = torchaudio.load(dir)
        resample = transforms.Resample(sample_rate, 8000)
        data = resample(data)
        data = self.mfcc(data)
        data = torch.mean(data, 2)
        data = data.to(self.device)
        logits = self.model(data)

        confidence = None
        label = None
        if logits[0] >= 6.886499881744385:
            label = 0
            confidence = torch.sigmoid(logits[0])
        else:
            label = 1
            confidence = 1 - torch.sigmoid(logits[0])
        return {"confidence": confidence.detach().item(), "label": label}


class SpAttackClTensor:
    def __init__(self):
        print("label 1: attacked")
        print("label 0: original")
        url = "https://drive.google.com/uc?id=14Gq-X4yhRFYq17hsUfC-WecVqPiMPIsc&export=download"
        r = requests.get(url, allow_redirects=True)
        open("sp-attack-cl.h5", "wb").write(r.content)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(
            "sp-attack-cl.h5",
            map_location=self.device,
        )
        self.mfcc = transforms.MFCC(
            sample_rate=8000,
            n_mfcc=256,
            norm="ortho",
            melkwargs={
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 256,
                "center": False,
            },
        )

    def __call__(self, data: torch.Tensor, sample_rate: int):
        resample = transforms.Resample(sample_rate, 8000)
        data = resample(data)
        data = self.mfcc(data)
        data = torch.mean(data, 2)
        data = data.to(self.device)
        logits = self.model(data)

        confidence = None
        label = None
        if logits[0] >= 6.886499881744385:
            label = 0
            confidence = torch.sigmoid(logits[0])
        else:
            label = 1
            confidence = 1 - torch.sigmoid(logits[0])
        return {"confidence": confidence.detach().item(), "label": label}

class WhisperClTensor:
    def __init__(self):
        print("label 1: attacked")
        print("label 0: original")
        url = "https://drive.google.com/uc?id=12GiFemoDlR2fZAwkpw6qIsvxRXO-ThZ6&export=download&confirm=t&uuid=69ad9228-9d85-47e1-832a-4bc8f0a7793a&at=AKKF8vyghulZ1pZlTJu3kBldbyWK:1683949951387 "
        r = requests.get(url, allow_redirects=True)
        open("WhisperCl.h5", "wb").write(r.content)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(
            "WhisperCl.h5",
            map_location=self.device,
        )
        self.whisper_model = WhisperModel.from_pretrained("openai/whisper-base")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        

    def __call__(self, data: torch.Tensor, sample_rate: int):
        resample = transforms.Resample(sample_rate, 16000)
        data = resample(data)
        inputs = self.feature_extractor(data[0], sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        decoder_input_ids = ((torch.tensor([[1, 1]]) * self.whisper_model.config.decoder_start_token_id)).to(self.device)
        logits = self.model(input_features, decoder_input_ids)

        confidence = None
        label = None
        if logits[0] >= 0.9041000008583069:
            label = 0
            confidence = torch.sigmoid(logits[0])
        else:
            label = 1
            confidence = 1 - torch.sigmoid(logits[0])
        return {"confidence": confidence.detach().item(), "label": label}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get the label of the audio")
    parser.add_argument(
        "-p", "--path", type=str, action="store", help="path to the audio file"
    )
    args = parser.parse_args()

    classifier = SpAttackCl()
    print(classifier(args.path))
