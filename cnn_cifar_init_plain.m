function net = cnn_cifar_init_plain(n, varargin)

opts.batchNormalization = true; 
opts = vl_argparse(opts, varargin); 

net = dagnn.DagNN();

% Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = [0.1*ones(1,80) 0.01*ones(1,40) 0.001*ones(1,40)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 128 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% First conv layer
add_block_conv(net, '0000', 'image', [3 3 3 16], 1, opts.batchNormalization); 

info.lastNumChannel = 16;
info.lastIdx = 0;

% 1st groups of layers
info = add_group(net, n, info, 3, 16, 1, opts.batchNormalization);

% 2nd groups of layers
info = add_group(net, n, info, 3, 32, 2, opts.batchNormalization);

% 3rd groups of layers
info = add_group(net, n, info, 3, 64, 2, opts.batchNormalization); 

% Prediction & loss layers
block = dagnn.Pooling('poolSize', [8 8], 'method', 'avg', 'pad', 0, 'stride', 1);
net.addLayer('pool_final', block, sprintf('relu%04d',info.lastIdx), 'pool_final');

block = dagnn.Conv('size', [1 1 info.lastNumChannel 10], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
lName = sprintf('fc%04d', info.lastIdx+1);
net.addLayer(lName, block, 'pool_final', lName, {[lName '_f'], [lName '_b']});

if opts.batchNormalization, % TODO confirm this is needed
  add_bn_layer(net, 10, lName, strrep(lName,'fc','bn'), 0.1); 
  lName = strrep(lName, 'fc', 'bn'); 
end

net.addLayer('softmax', dagnn.SoftMax(), lName, 'softmax');  
net.addLayer('loss', dagnn.Loss('loss', 'log'), {'softmax', 'label'}, 'loss');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'softmax','label'}, 'error') ;

net.initParams();

end

% Add a group of layers containing 2n conv layers
function info = add_group(net, n, info, w, ch, stride, bn)
% the 1st layer in the group downsample the activations by half
add_block_conv(net, sprintf('%04d', info.lastIdx+1), sprintf('relu%04d', info.lastIdx), ...
  [w w info.lastNumChannel ch], stride, bn); 
info.lastIdx = info.lastIdx + 1;
info.lastNumChannel = ch;
for i=2:2*n,
  add_block_conv(net, sprintf('%04d', info.lastIdx+1), sprintf('relu%04d', info.lastIdx), ...
    [w w ch ch], 1, bn);
  info.lastIdx = info.lastIdx + 1;
end
end

% Add a conv layer (followed by batch normalization & relu) 
function net = add_block_conv(net, out_suffix, in_name, f_size, stride, bn)
block = dagnn.Conv('size',f_size, 'hasBias',true, 'stride', stride, ...
                   'pad',[ceil(f_size(1)/2-0.5) floor(f_size(1)/2-0.5) ...
                   ceil(f_size(2)/2-0.5) floor(f_size(2)/2-0.5)]);
lName = ['conv' out_suffix];
net.addLayer(lName, block, in_name, lName, {[lName '_f'],[lName '_b']});
pidx = net.getParamIndex([lName '_b']);
net.params(pidx).weightDecay = 0;
if bn, 
  add_bn_layer(net, f_size(4), lName, strrep(lName,'conv','bn'), 0.1); 
  lName = strrep(lName, 'conv', 'bn');
end
block = dagnn.ReLU('leak',0);
net.addLayer(['relu' out_suffix], block, lName, ['relu' out_suffix]);
end

% Add a batch normalization layer
function net = add_bn_layer(net, n_ch, in_name, out_name, lr)
block = dagnn.BatchNorm('numChannels', n_ch);
net.addLayer(out_name, block, in_name, out_name, ...
  {[out_name '_g'], [out_name '_b'], [out_name '_m']});
pidx = net.getParamIndex({[out_name '_g'], [out_name '_b'], [out_name '_m']});
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(2)).weightDecay = 0; 
net.params(pidx(3)).learningRate = lr;
net.params(pidx(3)).trainMethod = 'average'; 
end

