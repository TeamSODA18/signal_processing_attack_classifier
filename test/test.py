from spClassify.infer import SpAttackCl
from spClassify.models import MLP

classifier = SpAttackCl()

print(classifier(r'test_audio\attack\33_after_clipping_23.wav'))
print(classifier(r'test_audio\original\common_voice_en_33164066.wav'))
