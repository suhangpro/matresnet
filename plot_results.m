function plot_results(expDir, savePath)
% Usage example: plot_results('exp', 'exp/summary.pdf');

if nargin==1, savePath = expDir; end
if strfind(savePath,'.pdf')==numel(savePath)-3, 
  savePath = fullfile(savePath,'summary.pdf');
end

plots = {'plain', 'resnet'}; 
figure(1) ; clf ;
cmap = lines;
for p = plots
  p = char(p) ;
  list = dir(fullfile(expDir,sprintf('cifar-%s-*',p)));
  tokens = regexp({list.name}, sprintf('cifar-%s-([\\d]+)',p), 'tokens'); 
  Ns = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens);
  Ns = sort(Ns); 
  subplot(1,numel(plots),find(strcmp(p,plots)));
  hold on;
  leg = {}; 
  Hs = [];
  for n=Ns,
    tmpDir = fullfile(expDir,sprintf('cifar-%s-%d',p,n));
    epoch = findLastCheckpoint(tmpDir);
    if epoch==0, continue; end
    load(fullfile(tmpDir,sprintf('net-epoch-%d.mat',epoch)),'stats');
    plot([stats.train.error]*100, ':','Color',cmap(find(Ns==n),:),'LineWidth',1.5); 
    Hs(end+1) = plot([stats.val.error]*100, '-','Color',cmap(find(Ns==n),:),'LineWidth',1.5); 
    leg{end+1} = sprintf('%s-%d',p,6*n+2);
  end
  xlabel('epoch') ;
  ylabel('error (%)');
  title(p) ;
  legend(Hs,leg{:},'Location','NorthEast') ;
%  axis square; 
%  ylim([0 25]);
  ylim([0 75]);
  xlim([1 160]);
  set(gca,'YGrid','on');
end
drawnow ;
print(1, fullfile(savePath,'summary.pdf'), '-dpdf') ;
end

function epoch = findLastCheckpoint(modelDir)
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
end
