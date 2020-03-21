# AVInstDisc

### Starter example

1) Training Cross-AVID with Nonlinear Head on Kinetics. (Batch size configured for at least 1 node)

``
python main-avid.py configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml
``

Naming conventions of AVID configs and model folders (eg Cross-N1024-NL.yaml)
 * Cross: Cross-modal instance discrimination
 * N1024: Number of negative memories
 * NL: Non-linear Head

2) Training CMA on top of previous run. (Batch size configured for at least 1 node)

``
python main-avid.py configs/main/avid-cma/r2plus1d/kinetics/NL-NCE-SelfX-N1024-PosW3-SelfConsensus-N64-Top32-Iter5.yaml
``

Naming conventions of CMA configs and model folders (eg NL-NCE-SelfX-N1024-PosW3-SelfConsensus-N64-Top32-Iter5.yaml)
 * NL: Non-Linear Head
 * SelfX: Cross Modal Instance Discrimination
 * N1024: Number of negatives memories
 * PosW: Within Modal Positive Set Discrimination
 * PosW3: 3 is the weight of PosDisc loss
 * SelfConsensus: Criterion for positive set computation
 * N64: Number of negatives in PosDisc loss
 * Top32: Size of positive set
 * Iter5: Number of epochs between positive set updates


3) Eval video network on UCF (Batch size configured for 2 gpus)

``
python eval-action-recg.py configs/benchmark/ucf/r2plus1d/r2plus1d-wucls-8at16-fold1.yaml configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml
``

4) Eval video network on Kinetics (Linear) (Batch size configured for 2 gpus)

``
python eval-action-recg-linear.py configs/benchmark/kinetics/r2plus1d/8x224x224-linear.yaml configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml
``

5) Eval audio network on ESC (SVM) (Batch size configured for 1 gpu)

``
python eval-snd-recg-svm.py configs/benchmark/esc50/audio-svm.yaml configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml
``

### Checkpoints and dataset meta-data

Checkpoints were backed-up in AWS (using fs3cmd utility). To restore checkpoints run

``
fs3cmd sync -p s3://fairusersglobal/users/pmorgado/h2/private/home/pmorgado/AVID-CMA/checkpoints ./{PROJECT_HOME}/checkpoints 
``

Dataset metadata was also backed-up in AWS.

``
fs3cmd sync -p s3://fairusersglobal/users/pmorgado/h2/private/home/pmorgado/AVID-CMA/datasets/cache/ ./{PROJECT_HOME}/datasets/cache/ 
``
