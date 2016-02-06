# matresnet
This is a Matlab (MatConvNet) implementation of ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385). 

### Install
```sh
git clone --recurse-submodules git@github.com:suhangpro/matresnet.git
cd matresnet/
matlab -nodisplay -r "setup(true,struct('enableGpu',true,'enableCudnn',true));exit;"
```

### Run experiments on Cifar10
```matlab
matlab>> run_cifar_experiments([9],'resnet','gpus',[1]);
```
![results](http://maxwell.cs.umass.edu/hsu/summary.png)
