# torch-classifier
This is a pytorch implementation of image classification in pytorch.  
You can use the models from the list belows:
- ResNet
- VGG
- Inception
- MobileNet
- ResNext
  
(Reference)  
- Code structure: [https://github.com/The-AI-Summer/Deep-Learning-In-Production](https://github.com/The-AI-Summer/Deep-Learning-In-Production)
- App: [https://github.com/jalalmansoori19/Cat-Classifier](https://github.com/jalalmansoori19/Cat-Classifier)

## Installation
1. Install docker. See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

2. Install torch-classifier
```bash
git clone https://github.com/ryoherisson/torch-classifier.git
```

3. Build docker image
```bash
cd torch-classifier
docker build \ 
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -f Dockerfile \
    -t {your-image-name}  \
    .
```

4. Run docker container
```bash
docker run -it --gpus {your-device} -v /home/{user}/torch-classifier:/workspace -p 8501:8501 --name {container-name} {your-image-name}
```

5. Setup
```bash
python setup.py develop --user
```

## Usage
Create a configuration file based on configs/default.yml.
```bash
### Configs ###

# Data Configs
data:
  dataroot: ./data/sample/images/
  labelroot: 
    train: ./data/sample/labels/train.csv
    val: ./data/sample/labels/test.csv
    test: ./data/sample/labels/test.csv
  img_size: [32, 32]
  n_channels: 3
  classes: ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
  color_mean: [0.4914, 0.4822, 0.4465]
  color_std: [0.2023, 0.1994, 0.2010]

# Training Configs
train:
  batch_size: 128
  epochs: 30
  optimizer:
    type: sgd
    lr: 0.001
    momentum: 0.9
    decay: 0.0001
  criterion:
    type: cross_entropy
  n_gpus: 1
  metrics: ['accuracy', 'precision', 'recall', 'f1score']
  save_ckpt_interval: 10
  resume:
    # e.g) resume: ./logs/2020-07-26T00:19:34.918002/ckpt/best_acc_ckpt.pth if resume. Blank if not resume

# Model Configs
model:
  name: resnet18
  n_classes: 10
  pretrained: False

# Other Configs
util:
  logdir: ./logs/
```

### Prepare Dataset
Prepare a directory with the following structure:
```bash
datasets/
├── images
│   ├── airplane1.png
│   ├── car1.png
│   ├── cat1.png
│   └── deer1.png
└── labels
    ├── train.csv
    └── test.csv
```

The content of the csv file should have the following structure.
```bash
filename,label
airplane1.png,0
car1.png,1
cat1.png,3
deer1.png,4
```

An example of a dataset can be found in the dataset folder.

### Train
```bash
python train.py --config ./configs/default.yml
```

### Evaluation
```bash
python train.py --config ./configs/default.yml --eval
```

### Tensorboard
```bash
tensorboard --logdir={logdir} --port={your-port}
```

### app
```bash
streamlit run app/app.py -- --configfile {your-config-path}
```

## Output of train and evaluation
You will see the following output in the log directory specified in the Config file.
```bash
logs/
└── 2020-07-26T14:21:39.251571
    ├── ckpt
    │   ├── best_acc_ckpt.pth
    │   ├── epoch0000_ckpt.pth
    │   └── epoch0001_ckpt.pth
    ├── metric
    │   ├── train_metric.csv
    │   ├── eval_metric.csv
    │   └── eval_cmx.png
    ├── tensorboard
    │   └── events.out.tfevents.1595773266.c47f841682de
    └── info.log
```

The contents of train_metric.csv and eval_metric.csv look like as follows:
```bash
epoch,train loss,train accuracy,train precision,train recall,train micro f1score
0,0.024899629971981047,42.832,0.4283200204372406,0.4215449392795563,0.42248743772506714
1,0.020001413972377778,54.61,0.5461000204086304,0.5404651761054993,0.5422631502151489
```
You will get loss, accuracy, precision, recall, micro f1score during training and as a result of evaluation.

The content of eval_cmx.png is a confusion matrix.