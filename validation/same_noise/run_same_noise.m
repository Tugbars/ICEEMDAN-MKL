% The authors' ICEEMDAN (ceemdan.m with emd.m, unchanged) on the library's
% noise realizations: randn.m in this folder hands them to ceemdan.m in place
% of random ones. Settings as in the validation: noise std 0.2, 100
% realizations, MaxIter 5000, SNRFlag 1.
%
% Needs in this folder: ceemdan.m, emd.m, randn.m, and for each signal
% <signal>.csv (one value per line) and noise_<N>.csv (100 rows of N, from
% "same_noise noise <N> 100 1 noise_<N>.csv").
%
% Writes same_<signal>.csv (the modes one per column, the residue last) and
% same_emd_<N>_<i>.csv (emd.m of realization i, i = 1..5, the way ceemdan.m
% decomposes its noise), each as soon as it is computed.

global SAME_NOISE SAME_NOISE_NEXT
sigs = {'burst', 'sines', 'baseline', 'wn'};
lengths_done = [];
for si = 1:numel(sigs)
  x = dlmread([sigs{si} '.csv'])';
  n = numel(x);
  SAME_NOISE = dlmread(sprintf('noise_%d.csv', n));
  if ~any(lengths_done == n)
    for i = 1:5
      modes = emd(SAME_NOISE(i, :));
      dlmwrite(sprintf('same_emd_%d_%d.csv', n, i), modes', 'precision', 17);
    end
    lengths_done(end + 1) = n;
  end
  SAME_NOISE_NEXT = 1;
  tic; modes = ceemdan(x, 0.2, 100, 5000, 1); t = toc;
  dlmwrite(sprintf('same_%s.csv', sigs{si}), modes', 'precision', 17);
  printf('%s: %d components, %.1f s, %d noise realizations used\n', sigs{si}, size(modes, 1), t, SAME_NOISE_NEXT - 1);
  fflush(stdout);
end
