# matresnet
This is a Matlab (MatConvNet) implementation of ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385). 

### Install
```sh
git clone --recurse-submodules git@github.com:suhangpro/matresnet.git
cd matresnet/
matlab -nodisplay -r "setup(true,struct('enableGpu',true,'enableCudnn',true));exit;"
```

### Experiments on Cifar10
```matlab
matlab>> % plain network w/ 20, 32, 44, 56 layers
matlab>> run_cifar_experiments([3 5 7 9], 'plain', 'gpus', [1]);
matlab>> % ResNet w/ 20, 32, 44, 56, 110 layers
matlab>> run_cifar_experiments([3 5 7 9 18], 'resnet', 'gpus', [1]);
```
![results](http://maxwell.cs.umass.edu/hsu/summary.png)
