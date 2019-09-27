function mrWriteLog( traj, filename )
    fid = fopen( filename, 'w' );
    for i = 1 : size( traj, 2 )
        mrWriteLogStruct( fid, traj(i).info, traj(i).inlier, traj(i).rt, traj(i).correspondence );
    end
    fclose( fid );
    disp( [ num2str( size( traj, 2 ) ), ' frames have been written.' ] );
end

function mrWriteLogStruct( fid, info, inlier, rt, correspondence )
    fprintf( fid, '%d\t%d\t%d\t%d\t%d\t%f\n', info(1), info(2), info(3), inlier(1), inlier(2), inlier(3) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\n', rt(1,1), rt(1,2), rt(1,3), rt(1,4) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\n', rt(2,1), rt(2,2), rt(2,3), rt(2,4) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\n', rt(3,1), rt(3,2), rt(3,3), rt(3,4) );
    
    for i=1:1:size(correspondence, 1)
        fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n', ...
            correspondence(i,1), correspondence(i,2), correspondence(i,3), ...
            correspondence(i,4), correspondence(i,5), correspondence(i,6), correspondence(i,7) );
    end
end
