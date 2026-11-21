function r = randn(varargin)
% Stands in for Octave's and MATLAB's randn in this folder only: hands
% ceemdan.m the library's noise realizations, one per call, in order (see
% run_same_noise.m). Any other use of randn here is an error.
global SAME_NOISE SAME_NOISE_NEXT
if isempty(SAME_NOISE) || nargin ~= 1 || ~isequal(varargin{1}, [1, size(SAME_NOISE, 2)])
  error('randn.m (validation/same_noise): only randn(size(x)) after run_same_noise.m set the noise');
end
r = SAME_NOISE(SAME_NOISE_NEXT, :);
SAME_NOISE_NEXT = SAME_NOISE_NEXT + 1;
end
