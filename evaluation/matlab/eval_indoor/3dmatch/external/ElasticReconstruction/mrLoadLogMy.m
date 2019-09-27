function [ traj ] = mrLoadLogMy( filename )
    fid = fopen( filename );
    k = 1;
    x = fscanf( fid, '%d', [1 3] );
    while ( size( x, 2 ) == 3 )
        m = fscanf( fid, '%f', [4 4] );
        inlierData = fscanf(fid, '%d\t%f', [1, 2]);
        inlierNum = inlierData(1);
        inlierRatio = inlierData(2);
        information = fscanf( fid, '%f', [6 6] );
        traj( k ) = struct( 'info', x, 'trans', m', 'inlierNum', inlierNum, 'inlierRatio', inlierRatio, 'information', information );
        k = k + 1;
        x = fscanf( fid, '%d', [1 3] );
    end
    fclose( fid );
    %disp( [ num2str( size( traj, 2 ) ), ' frames have been read.' ] );
end
