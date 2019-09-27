function test_dataset = read_txts_kitti(root)
% test_dataset: [seq, idx1, idx2, t1, t2, t3, qw, qx, qy, qz]

test_dataset = [];
for seq = 0:1:10
    seq_str = sprintf('%02d', seq);
    txt_path = fullfile(root, seq_str, 'groundtruths.txt');
    txt_content = dlmread(txt_path, '\t', 1, 0);
    
    seq_column = zeros(size(txt_content, 1), 1) + seq;
    
    test_dataset = [test_dataset;[seq_column, txt_content]];
end

end