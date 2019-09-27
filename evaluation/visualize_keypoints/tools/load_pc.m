function pc = load_pc(filepath)
% determine file type
typename = filepath(end-2:end);
if strcmp(typename, 'npy')
    pc = readNPY(filepath);
elseif strcmp(typename, 'bin')
    pc = Utils.loadPointCloud(filepath, 6);
else
    fprintf('error loading pc %s', filepath);
end

end