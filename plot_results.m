function plot_results(expDir, datasetName, savePath)
% Usage example: plot_results('exp', 'cifar', 'exp/summary.pdf');

if nargin<=2, savePath = expDir; end
if nargin<=1, datasetName = 'cifar'; end
if isempty(strfind(savePath,'.pdf')) || strfind(savePath,'.pdf')~=numel(savePath)-3, 
  savePath = fullfile(savePath,'cifar-summary.pdf');
end

plots = {'plain', 'resnet'}; 
figure(1) ; clf ;
cmap = lines;
for p = plots
  p = char(p) ;
  list = dir(fullfile(expDir,sprintf('%s-%s-*',datasetName,p)));
  tokens = regexp({list.name}, sprintf('%s-%s-([\\d]+)',datasetName,p), 'tokens'); 
  Ns = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens);
  Ns = sort(Ns); 
  subplot(1,numel(plots),find(strcmp(p,plots)));
  hold on;
  leg = {}; Hs = []; nEpoches = 0;
  for n=Ns,
    tmpDir = fullfile(expDir,sprintf('%s-%s-%d',datasetName,p,n));
    epoch = findLastCheckpoint(tmpDir);
    if epoch==0, continue; end
    load(fullfile(tmpDir,sprintf('net-epoch-%d.mat',epoch)),'stats');
    plot([stats.train.error]*100, ':','Color',cmap(find(Ns==n),:),'LineWidth',1.5); 
    Hs(end+1) = plot([stats.val.error]*100, '-','Color',cmap(find(Ns==n),:),'LineWidth',1.5); 
    leg{end+1} = sprintf('%s-%d',p,6*n+2);
    if numel(stats.train)>nEpoches, nEpoches = numel(stats.train); end
  end
  xlabel('epoch') ;
  ylabel('error (%)');
  title(p) ;
  legend(Hs,leg{:},'Location','NorthEast') ;
%  axis square; 
%  ylim([0 25]);
  ylim([0 75]);
  xlim([1 nEpoches]);
  set(gca,'YGrid','on');
end
drawnow ;
print(1, savePath, '-dpdf') ;
end

function epoch = findLastCheckpoint(modelDir)
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
end
