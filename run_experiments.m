function run_experiments(Ns, MTs, varargin)
% Usage example: run_experiments([3 5 7 9], 'plain', 'gpus', [1]); 
% Options: 
%   'expDir'['exp'], 'bn'[true], 'gpus'[[]], 'border'[[4 4 4 4]], 
%   'meanType'['image'], 'whitenData'[true], 'contrastNormalization'[true]
%   and more defined in cnn_cifar.m

setup;

opts.expDir = 'exp';
opts.bn = true;
opts.meanType = 'image';
opts.whitenData = true;
opts.contrastNormalization = true; 
opts.border = [4 4 4 4];
opts.gpus = [];

opts = vl_argparse(opts, varargin); 

n_exp = numel(Ns); 
if ischar(MTs) || numel(MTs)==1, 
  if ischar(MTs), MTs = {MTs}; end; 
  MTs = repmat(MTs, [1, n_exp]); 
else
  assert(numel(MTs)==n_exp);
end

expRoot = opts.expDir; 
for i=1:n_exp, 
  opts.expDir = fullfile(expRoot, ...
    sprintf('cifar-%s-%d', MTs{i}, Ns(i))); 
  [net,info] = cnn_cifar(Ns(i), 'modelType', MTs{i}, opts); 
end
