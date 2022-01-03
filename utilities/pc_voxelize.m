function [pcOut] = pc_voxelize(pcIn, voxelDepthOut, isVoxelized)
% Copyright (C) 2020 ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
%
%     Multimedia Signal Processing Group (MMSPG)
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%
% Author:
%   Evangelos Alexiou (evangelos.alexiou@epfl.ch)
%
% Reference:
%   Evangelos Alexiou, Kuan Tung, Touradj Ebrahimi, "Towards neural network
%   approaches for point cloud compression," Proc. SPIE 11510, Applications
%   of Digital Image Processing XLIII, 1151008
%
% Quantization of a point cloud's coordinates to a regular grid of voxels.
% The indices of the grid are non-negative integers that span in the range
% [0, 2^voxelDepth-1]. The color assigned to each voxel is obtained after
% averaging the colors of points that fall in the corresponding voxel.
%
%   [pcOut] = pc_voxelize(pcIn, voxelDepthOut, isVoxelized)
%
%   INPUTS
%       pcIn:           The input point cloud, as a pointCloud object.
%       voxelDepthOut:  Non-negative integer that defines the depth of 
%                       the output voxel grid.
%       isVoxelized:    Boolean flag to indicate if the point cloud is
%                       already voxelized or not.
%
%   OUTPUTS
%       pcOut:          The output voxelized point cloud, as a pointCloud 
%                       object.
%
%   Example:
%   [ptCloudOut] = pc_voxelize(ptCloudIn, 10, false)


if nargin < 3
    error('Too few input arguments.');
else
    ptCloud = pcIn;
end


% Convert geometry and color to double
geomIn = double(ptCloud.Location);
if ~isempty(ptCloud.Color)
    clrsIn = double(ptCloud.Color);
end


% Uniform quantization   
if ~(isVoxelized)
    % Normalize geometry to range [0, 1]
    geomIn = geomIn - min(geomIn,[],1);
    geomIn = geomIn./max(geomIn(:));

    % Quantization step
    quantStep = 1 / (2^(voxelDepthOut) - 1);

    % Quantized geometry in grid range of [0, 2^N - 1]
    quantGeom = floor(geomIn/quantStep + 1/2);

else
    % Get current voxel depth
    curVoxDepth = ceil(log2(max([max(geomIn) - min(geomIn)]))); 
     
    % Normalize to a sub-interval that corresponds to the occupied voxels,
    % so the original distribution of geometry over the grid is maintained.
    geomIn = geomIn./(2^curVoxDepth - 1);      
    
    % Quantization step
    quantStep = 1 / (2^voxelDepthOut - 1);

    % Quantized geometry in grid range of [0, 2^N - 1]
    quantGeom = floor(geomIn/quantStep + 1/2);
    
    % Shift model, if voxels fall outside of the grid
    if sum(min(quantGeom) < 0) > 0 || sum(max(quantGeom) > 2^10) > 0
        quantGeom = quantGeom - min(quantGeom);
    end
        
end


% Sort quantized geometry (first sort last column, then second, then first)
[quantGeomSorted, col] = sortrows(quantGeom);
if ~isempty(ptCloud.Color)
    % Sort color attributes accordingly
    clrsSorted = clrsIn(col, :);
end


% Find indexes of points with different quantized geometry
d = diff(quantGeomSorted,1,1);
sd = sum(abs(d),2) > 0;
id = [1; find(sd == 1)+1];

if ~isempty(ptCloud.Color)
    % Add threshold index for last entries
    id = [id; size(quantGeomSorted,1)+1];
    
    % Average color of points with same quantized geometry
    clrsBlend = zeros(size(id,1)-1,3);
    for j = 1:size(id,1)-1
        clrsBlend(j,:) = mean(clrsSorted(id(j):id(j+1)-1, :), 1);
    end
    
    % Remove threshold index
    id(end) = [];
    
    % Voxel coordinates as unique quantized geometry
    geomOut = single(quantGeomSorted(id,:));
    
    % Voxel colors as blend of color values of points with same
    % quantized geometry
    clrOut = uint8(round(clrsBlend));

    % Voxelized point cloud
    pcOut = pointCloud(geomOut, 'Color', clrOut);

else
    % Voxel coordinates as unique quantized geometry
    geomOut = single(quantGeomSorted(id,:));

    % Voxelized point cloud
    pcOut = pointCloud(geomOut);
end
