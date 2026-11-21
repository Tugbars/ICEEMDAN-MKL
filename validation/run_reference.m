% Run the authors' ICEEMDAN (ceemdan.m, with Rilling & Flandrin's emd.m) on
% the validation signals: noise std 0.2, 100 realizations, MaxIter 5000,
% SNRFlag 1 (beta_k = eps * std(r_k)), seeds 1..3.
sigs = {'wn', 'sines'};
for si = 1:numel(sigs)
  x = dlmread([sigs{si} '.csv'])';
  for s = 1:3
    randn('seed', s); rand('seed', s);
    tic; modes = ceemdan(x, 0.2, 100, 5000, 1); t = toc;
    dlmwrite(sprintf('ref_%s_%d.csv', sigs{si}, s), modes', 'precision', 17);
    printf('%s seed %d: %d components, %.1f s\n', sigs{si}, s, size(modes, 1), t); fflush(stdout);
  end
end
