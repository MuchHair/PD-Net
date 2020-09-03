# PD-NET(ECCV2020) 

### Experiment Results:
#### HICO-DET
| model     |  mAP | 
|--------------|-----------|
|Baseline |  | 
|PD        |  |

#### V-COCO
| model     |  mAP | 
|--------------|-----------|
|Baseline | 48.2 | 
|PD        | 51.6 |
|PD+    | 52.6 | 

### Setup  Environment:
see [No_frills](https://github.com/BigRedT/no_frills_hoi_det) for details
##### build for vcoco
```
conda install pycocotools -c conda-forge
```

### Prepare Data:
#### HICO-DET

#### V-COCO
##### Annotations
Download all annotations from [here](https://pan.baidu.com/s/1Z4aHLV9GMdZ3XdEFFqf4yg) psw:b4il and put them into `data/vcoco/annotations/`
##### Training,val,test data(Generate or Download(Recommend!!!))
##### Generate
1)run faster-rcnn
```
# prepare input for faster-rcnn
python -m lib.data_process_vcoco.prepare_data_for_faster_rcnn
```
use [faster-rcnn](https://github.com/SherlockHolmes221/pytorch-faster-rcnn) to get bbox

```
# select and generate candidates
python -m lib.data_process.select
python -m lib.data_process.hoi_candidates
```
2)get human,object,union features

use [faster-rcnn](https://github.com/SherlockHolmes221/pytorch-faster-rcnn) to get features

3)run AlphaPose 
```
# prepare input file for AlphaPose
python -m lib.data_process.prepare_for_pose
```
use  [AlphaPose](https://github.com/SherlockHolmes221/AlphaPose) to get keypoints
```
# convert and generate features
python -m lib.data_process_hico.cpn_convert
python -m lib.data_process.cache_alphapose_features
```
4)bbox
```
python -m lib.data_process.cache_box_features
```
5)labels
```
python -m lib.data_process.label_hoi_candidates
```
6)nis
```
python -m lib.data_process.nis_hoi_candidates
```
#### Download 
Download training,val,test data from [here](https://pan.baidu.com/s/16VO33ac1IFKkO0dSLEFYEQ) psw:245d and put them into `data/vcoco/`
Download v-coco eval data from [here](/home/xian/Documents/code/PD-Net/data/vcoco) psw:cws6 and put them into `eval/data/`

### Train, Test and Eval Model
##### Baseline
```
# train
CUDA_VISIBLE_DEVICES=1 python tools/vcoco/train_net_baseline.py

# test (use tensorboard to choose the best model and the precoss will generate a pickle file used for eval)
CUDA_VISIBLE_DEVICES=0 python tools/vcoco/test_net_baseline.py

# eval 
python eval/eval_example.py --file pickle_file
```

##### PD
```
# train
CUDA_VISIBLE_DEVICES=0 python tools/vcoco/train_net_pd.py

# test(use tensorboard to choose the best model and the precoss will generate a pickle file used for eval)
CUDA_VISIBLE_DEVICES=0 python tools/vcoco/test_net_pd.py

# eval 
python eval/eval_example.py --file pickle_file
```

##### INet
```
# train
CUDA_VISIBLE_DEVICES=0 python tools/vcoco/train_nis.py

# test (use tensorboard to choose the base model)
CUDA_VISIBLE_DEVICES=0 python tools/vcoco/test_nis.py
 
# generate a pickle file(remember to change the model dir in eval/generate_pkl_nis.py)
python -m eval.generate_pkl_nis

# eval
python eval/eval_example.py --file pickle_file
```
### Trained Model:
#### V-COCO
Download trained model from [here](https://pan.baidu.com/s/1IchSpsVrBV7ByVCx3kRw7Q) psw: u6os



