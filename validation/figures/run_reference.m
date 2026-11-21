% The authors' ICEEMDAN (ceemdan.m with Rilling & Flandrin's emd.m) on the signals of the README
% figure and timings: noise std 0.2, 100 realizations, MaxIter 5000, SNRFlag 1 (the improved
% version). Runs in GNU Octave and in MATLAB, from a folder that holds ceemdan.m, emd.m and the
% CSV files of make_signals.py.
%
% Writes ref_burst_<seed>.csv (modes one per column, residue last) and ref_times.csv
% (signal, n, run, seconds, components), one line as soon as each decomposition ends.

fid = fopen('ref_times.csv', 'w');
fprintf(fid, 'signal,n,run,seconds,components\n');
fclose(fid);

% the test signal of the figure, seeds 1..3
x = dlmread('burst.csv')';
for s = 1:3
  randn('seed', s); rand('seed', s);
  tic; modes = ceemdan(x, 0.2, 100, 5000, 1); t = toc;
  dlmwrite(sprintf('ref_burst_%d.csv', s), modes', 'precision', 17);
  fid = fopen('ref_times.csv', 'a');
  fprintf(fid, 'burst,%d,%d,%.3f,%d\n', numel(x), s, t, size(modes, 1));
  fclose(fid);
end

% time per decomposition against signal length
lengths = [1024 2048 4096 8192];
for li = 1:numel(lengths)
  x = dlmread(sprintf('tones_%d.csv', lengths(li)))';
  for r = 1:2
    randn('seed', r); rand('seed', r);
    tic; modes = ceemdan(x, 0.2, 100, 5000, 1); t = toc;
    fid = fopen('ref_times.csv', 'a');
    fprintf(fid, 'tones,%d,%d,%.3f,%d\n', lengths(li), r, t, size(modes, 1));
    fclose(fid);
  end
end
