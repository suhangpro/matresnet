function run_experiments(Ns, MTs, gpus)
% usage example: run_experiments([3 5 7 9], 'plain', [1]); 

setup;

opts.meanType = 'image';
opts.whitenData = true;
opts.contrastNormalization = true; 
opts.border = [4 4 4 4];
opts.gpus = gpus;


n_exp = numel(Ns); 
if ischar(MTs) || numel(MTs)==1, 
  if ischar(MTs), MTs = {MTs}; end; 
  MTs = repmat(MTs, [1, n_exp]); 
else
  assert(numel(MTs)==n_exp);
end

for i=1:n_exp, 
  [net,info] = cnn_cifar(Ns(i), 'modelType', MTs{i}, opts); 
end
