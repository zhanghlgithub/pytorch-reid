#!/usr/bin/env python3
import torchreid

#2 Load data manager
datamanager = torchreid.data.ImageDataManager(
    root='./data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=16,
    transforms=['random_flip', 'random_crop']
)

#3 Build model, optimizer and lr_scheduler
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

#4 Build engine
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

#5 Run training and test
engine.run(
    save_dir='log/osnet_x1_0',
    max_epoch=5,
    eval_freq=1,
    print_freq=10,
    test_only=True,
    visrank=False
)

