# Pseudo Labels Guided Pixels Contrast for Domain Adaptive Semantic Segmentation
by Jianzi Xiang, Cailu Wan, Zhu Cao.

## Overview
Semantic segmentation is essential for comprehending images, but the process necessitates a substantial amount of detailed
annotations at the pixel level. Acquiring such annotations can be costly in the real-world. Unsupervised domain adaptation
(UDA) for semantic segmentation is a technique that uses virtual data with labels to train a model and adapts it to real data
without labels. Some recent works use contrastive learning, which is a powerful method for self-supervised learning, to help
with this technique. However, these works do not take into account the diversity of features within each class when using
contrastive learning, which leads to errors in class prediction. We analyze the limitations of these works and propose a novel
framework called Pseudo-label Guided Pixel Contrast (PGPC), which overcomes the disadvantages of previous methods. We
also investigate how to use more information from target images without adding noise from pseudo-labels. We test our method on
two standard UDA benchmarks and show that it outperforms existing methods. Specifically, we achieve relative improvements
of 5.1% mIoU and 4.6% mIoU on the Grand Theft Auto V (GTA5) to Cityscapes and SYNTHIA to Cityscapes tasks based on
DAFormer, respectively. Furthermore, our approach can enhance the performance of other UDA approaches without increasing
model complexity

## Setup Environment
We used python 3.8.5:

    conda create -n pgpc python == 3.8.5
    conda activate pgpc

and we should install the requirements:

    conda install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    conda install mmcv-full==1.3.7  # requires the other packages to be installed first

## Setup Dataset
**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to data/cityscapes.

**GTA:** Please, download all image and label packages from [here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to data/gta.

**Synthia (Optional):** Please, download SYNTHIA-RAND-CITYSCAPES from [here](http://synthia-dataset.net/downloads/) and extract it to data/synthia.

**ACDC (Optional):** Please, download rgb_anon_trainvaltest.zip and gt_trainval.zip from [here](https://acdc.vision.ee.ethz.ch/download) and extract them to data/acdc. Further, please restructure the folders from condition/split/sequence/ to split/ using the following commands:

    rsync -a data/acdc/rgb_anon/*/train/*/* data/acdc/rgb_anon/train/
    rsync -a data/acdc/rgb_anon/*/val/*/* data/acdc/rgb_anon/val/
    rsync -a data/acdc/gt/*/train/*/*_labelTrainIds.png data/acdc/gt/train/
    rsync -a data/acdc/gt/*/val/*/*_labelTrainIds.png data/acdc/gt/val/

**Dark Zurich (Optional):** Please, download the Dark_Zurich_train_anon.zip and Dark_Zurich_val_anon.zip from [here](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) and extract it to data/dark_zurich.

## Training
For example, We can run the following code for training PGPC + Daformer on Gta -> Cityscapes task：

    python3 run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py

## Testing

    python -m tools.test path/to/config_file path/to/checkpoint_file --test-set --format-only --eval-option imgfile_prefix=labelTrainIds to_label_id=False
