# AVInstDisc

### Starter example

* Training Cross-AVID with Nonlinear Head on Kinetics. (Batch size configured for at least 4 node)

    ``
    python main-avid.py configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml
    ``

    Naming conventions of AVID configs and model folders
     * Example: `Cross-N1024-NL.yaml`
     * `Cross`: Cross-modal instance discrimination
     * `N1024`: Number of negative memories
     * `NL`: Non-linear Head

* Training CMA on top of previous run. (Batch size configured for at least 4 node)

    ``
    python main-avid.py configs/main/avid-cma/r2plus1d/kinetics/NL-NCE-SelfX-N1024-PosW3-SelfConsensus-N64-Top32-Iter5.yaml
    ``

    Naming conventions of CMA configs and model folders
     * Example: `NL-NCE-SelfX-N1024-PosW3-SelfConsensus-N64-Top32-Iter5.yaml`
     * `NL`: Non-Linear Head;
     * `SelfX`: Cross Modal Instance Discrimination;
     * `N1024`: Number of negatives memories;
     * `PosW3`: Within Modal Positive Set Discrimination with a loss weight of 3;
     * `SelfConsensus`: Criterion for positive set computation;
     * `N64`: Number of negatives in PosDisc loss;
     * `Top32`: Size of positive set;
     * `Iter5`: Number of epochs between positive set updates;
     * Model folder specified in the config.

* Eval video network on UCF (Batch size configured for 2 gpus)

    ``
    python eval-action-recg.py configs/benchmark/ucf/r2plus1d/r2plus1d-wucls-8at16-fold1.yaml configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml
    ``

* Eval video network on Kinetics (Linear) (Batch size configured for 2 gpus)

    ``
    python eval-action-recg-linear.py configs/benchmark/kinetics/r2plus1d/8x224x224-linear.yaml configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml
    ``

* Eval audio network on ESC (SVM) (Batch size configured for 1 gpu)

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

### Experiments
Config files used for each experiment.

* AVID
  * Audioset-100k
    * Non-Linear Head 
      - `configs/main/avid/r2plus1d-small/audioset-100k/Cross-N1024-NL.yaml`
      - `configs/main/avid/r2plus1d-small/audioset-100k/Joint-N1024-NL.yaml`
      - `configs/main/avid/r2plus1d-small/audioset-100k/Self-N1024-NL.yaml`
    * Linear Head 
      - `configs/main/avid/r2plus1d-small/audioset-100k/Cross-N1024.yaml`
      - `configs/main/avid/r2plus1d-small/audioset-100k/Joint-N1024.yaml`
      - `configs/main/avid/r2plus1d-small/audioset-100k/Self-N1024.yaml`
    * Kinetics
      - `configs/main/avid/r2plus1d/kinetics/Cross-N1024.yaml`
      - `configs/main/avid/r2plus1d/kinetics/Cross-N1024-NL.yaml`
    * Audioset
      - `configs/main/avid/r2plus1d/audioset/Cross-N1024.yaml`
      - `configs/main/avid/r2plus1d/audioset/Cross-N1024-NL.yaml`
      
* CMA
  *  