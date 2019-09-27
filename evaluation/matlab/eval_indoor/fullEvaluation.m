clear all;
close all;

% using lite evaluation!!!
% 1. ransac 1000 iterations
% 2. only evluate on overlapped pairs
% 3. use fake ratioAligned

runFragmentRegistration
writeLog
evaluate 
% split_txt