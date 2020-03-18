# AVInstDisc

Starter example

1) Training Cross-AVID with Nonlinear Head on Kinetics. (Batch size configured for at least 1 node)
python main-avid.py configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml

2) Training CMA on top of previous run. (Batch size configured for at least 1 node)
python main-avid.py configs/main/avid-cma/r2plus1d/kinetics/NL-NCE-SelfX-N1024-PosW3-SelfConsensus-N64-Top32-Iter5.yaml

3) Eval video network on UCF (Batch size configured for 2 gpus)
python eval-action-recg.py configs/benchmark/ucf/r2plus1d/r2plus1d-wucls-8at16-fold1.yaml configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml

4) Eval video network on Kinetics (Linear) (Batch size configured for 2 gpus)
python eval-action-recg-linear.py configs/benchmark/kinetics/r2plus1d/8x224x224-linear.yaml configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml

5) Eval audio network on ESC (SVM) (Batch size configured for 1 gpu)
python eval-snd-recg-svm.py configs/benchmark/esc50/audio-svm.yaml configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml
