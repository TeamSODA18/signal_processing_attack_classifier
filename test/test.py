from spClassify.infer import SpAttackCl, SpAttackClTensor, WhisperClTensor, SpMelCl
from spClassify.models import MLP, WhisperCl, CNN
import torchaudio

# classifier = SpAttackCl()

# print(classifier(r'test_audio\attack\33_after_clipping_23.wav'))
# print(classifier(r'test_audio\original\common_voice_en_33164066.wav'))


# classifier = SpAttackClTensor()

# data, sample_rate = torchaudio.load(r'test_audio\attack\33_after_clipping_23.wav')

# print(type(data))
# print(classifier(data, sample_rate))

# classifier = WhisperClTensor()

# print(classifier(data, sample_rate))

# data, sample_rate = torchaudio.load(r'test_audio\original\common_voice_en_33164066.wav')

# print(classifier(data, sample_rate))

classifier = SpMelCl()

data, sample_rate = torchaudio.load(r'test_audio/attack/33_after_clipping_23.wav')

print(classifier(data, sample_rate))