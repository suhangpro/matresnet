function net = cnn_cifar_init_plain(n, varargin)

opts.batchNormalization = true; 
opts = vl_argparse(opts, varargin); 

net = dagnn.DagNN();

% Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = [0.1*ones(1,80) 0.01*ones(1,40) 0.001*ones(1,40)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
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

% TODO confirm bn is necessary/not harmful for the last fc layer
if opts.batchNormalization, 
  lName = strrep(lName, 'fc', 'bn'); 
  net.addLayer(lName, dagnn.BatchNorm('numChannels',10), strrep(lName,'bn','fc'), lName, ...
    {[lName '_g'], [lName '_b'], [lName '_m']});
end

net.addLayer('softmax', dagnn.SoftMax(), lName, 'softmax');  

net.addLayer('loss', dagnn.Loss('loss', 'log'), {'softmax', 'label'}, 'loss');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'softmax','label'}, 'error') ;

end

% Add a group of layers containing 2n conv layers
function info = add_group(net, n, info, w, ch, stride, bn)

% the 1st layer in the group downsample the responses by half
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

if bn, 
  block = dagnn.BatchNorm('numChannels', f_size(4));
  lName = strrep(lName, 'conv', 'bn');
  net.addLayer(lName, block, ['conv' out_suffix], lName, ...
    {[lName '_g'], [lName '_b'], [lName '_m']});
end

block = dagnn.ReLU('leak',0);
net.addLayer(['relu' out_suffix], block, lName, ['relu' out_suffix]);

end

