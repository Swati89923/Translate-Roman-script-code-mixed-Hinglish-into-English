| Rank | Model | Trained by | How | BLEU | chrF |
| --- | --- | --- | --- | --- | --- |
| 1 | mBART-50 fine-tuned | Vinay | pretrained + fine-tuned | 36.76 | 57.87 |
| 2 | Transformer, 6 layers (beam 5) | Ujjval | from scratch | 28.51 | 50.32 |
| 3 | mBART-50 zero-shot | Vinay | pretrained, no training | 22.14 | 38.35 |
| 4 | mT5-small fine-tuned | Vinay | pretrained + fine-tuned | 18.97 | 33.79 |
| 5 | LSTM + Bahdanau attention (beam 5) | Pinky | from scratch | 8.97 | 27.71 |
| 6 | mT5-small zero-shot | Vinay | pretrained, no training | 0.01 | 2.33 |
