function aststr = FonaDynPlotVRPdiffs(sArray1, sArray2, colNames, colName, fig, varargin)
%% function FonaDynPlotVRPdiffs(sArray1, sArray2, colNames, colName, fig, ...)
% Given two input VRPs, plot a VRP in which each cell has the value
% of the difference between the corresponding cells in the input VRPs, for assessing
% changes. 
% <sArray> are arrays of numbers and <colNames> is a cell array of column names, 
% both previously returned by FonaDynLoadVRP.m. 
% <colName> is the name of the column to be plotted (case sensitive).
% <fig> is the number of the current figure or subplot.
% Optional arguments: 
% 'MinCycles', integer       - set a minimum cycles-per-cell threshold
% 'Range', [foMin, foMax, Lmin, Lmax]   - specify plot range
% 'OffsetSPL', value         - offset the dB SPL (for calibration)
% 'ColorBar', 'on' or 'off'  - show a horizontal color bar at the top left

minCycles = 1;
xmax = 91;  % can be <=96
xmin = 32;  % can be >=30
ymax = 120; % can be <=120
ymin = 30;  % can be >=40
offsetSPL = 0;  % useful for SPL re-calibration
plotHz = 0;
diffRange = [-10, 10];
conLevels = [-0.1:0.05:0.1];
bMesh = 0;
bColorBar = 0;
cbLabel = 'Difference';
tickLabels = {}; 
ticks = [];
bOverlay = 0;
bFlip = 1;

args = nargin-5;

% Handle any optional arguments
for i = 1 : 2 : args
    switch varargin{i}
        case 'MinCycles'
            minCycles = varargin{i+1};
        case 'Range'
            range = varargin{i+1};
            xmin = range(1);
            xmax = range(2);
            ymin = range(3);
            ymax = range(4);
        case 'OffsetSPL'
            offsetSPL = varargin{i+1};
        case 'ColorBar'
            if strcmpi('on', varargin{i+1})
                bColorBar = 1;
            end
        case 'PlotHz'
            if strcmpi('on', varargin{i+1})
                plotHz = 1;
            end
        case 'Mesh'
            if strcmpi('on', varargin{i+1})
                bMesh = 1;
            end
        case 'Region'
            overArray = varargin{i+1};
            if size(overArray, 1) > 0
               bOverlay = 1;
            end
        case 'DiffRange'
            diffRange = varargin{i+1};
        otherwise
            warning (['Unrecognized option: ' varargin{i}]);
    end
end

colPlot = find(ismember(colNames, colName)); 
if size(colPlot,2) < 1
    warning(['Unrecognized column name: ' colName]);
    return
end
diff_min = min(sArray1(:,colPlot)-sArray2(:,colPlot));
diff_max = max(sArray1(:,colPlot)-sArray2(:,colPlot));
abs_max = max(abs(diff_min), abs(diff_max));
% Adjust the range based on the signs of min and max
if diff_min < 0
    newMin = -abs_max;
else
    newMin = 0;
end

if diff_max < 0
    newMax = 0;
else
    newMax = abs_max;
end
diffRange = [newMin, newMax];
colMaxCluster = find(ismember(colNames, 'maxCluster')); 
colTotal = find(ismember(colNames, 'Total')); 
nClusters = size (sArray1, 2) - colMaxCluster;


% axH = gca;

% Set up a colormap from red to green through near-white
%[colors, cmin, cmax] = FonaDynColors(colName, nClusters);
R1 = linspace(0.97, 0.97, 32); 
R2 = linspace(0.97, 0.0, 32);
G1 = linspace(0.0, 0.97, 32); 
G2 = linspace(0.97, 0.97, 32); 
B1 = linspace(0, 0.97, 32); 
B2 = linspace(0.97, 0, 32);
colors = [ [R1 R2] ; [G1 G2]; [B1 B2]]';
cmin = diffRange(1);
cmax = diffRange(2);
% cmin = -0.2;
% cmax =  0.2;

dudNum = -99999;
pix1 = ones(150,100)*dudNum;
pix2 = ones(150,100)*dudNum;
allPixels = ones(150,100)*dudNum;

indices = find(sArray1(:, colTotal) >= minCycles);
for i=1:length(indices)
    y = sArray1(indices(i), 2) + round(offsetSPL);
    x = sArray1(indices(i), 1);
    z = sArray1(indices(i), colPlot);
    pix1(y, x) = z;
end

indices = find(sArray2(:, colTotal) >= minCycles);
for i=1:length(indices)
    y = sArray2(indices(i), 2) + round(offsetSPL);
    x = sArray2(indices(i), 1);
    z = sArray2(indices(i), colPlot);
    pix2(y, x) = z;
end

% Compute the per-element differences pix2-pix1
% 
diffs = ones(1);
diffIx = 1;
for x = xmin : xmax
    for y = ymin : ymax
        value1 = pix1(y, x);
        value2 = pix2(y, x); 
        if (value1 ~= dudNum) & (value2 ~= dudNum)
            % if value1*value2 > 0
                diffs(diffIx) = value2 - value1;
                allPixels(y, x) = value2 - value1;
                diffIx = diffIx + 1; 
%                allPixels(y, x) = value2 - value1;
%             else
%                 allPixels(y, x) = 1.0;
            %end
        end
    end
end

aststr = '';
[h,p] = ttest(diffs);
mdiff = mean(diffs);
diffstr = num2str(mdiff,3);
aststr = diffstr;
if h > 0
      if p <= 0.01
        aststr = [diffstr '*' ];
        if p <= 0.001
            aststr = [diffstr '**' ];
            if p <= 0.0001
                aststr = [diffstr '***'];
            end
        end
    end
end

cbLabel = ['Mean diff: ' aststr];

% Plot the cells astride the grid, not next to it
X = (1:100)-0.5;
Y = (1:150)-0.5;
%handle = pcolor(X,Y,allPixels);

% cFloor = 0.5*min(min(allPixels));
% cCeil  = 0.5*max(max(allPixels));
% cmax = max(abs(cFloor), abs(cCeil));
% cmin = -cmax;

% figure(fig);
axH = gca;
grid on

if (bMesh == 0)
    duds = find(allPixels==dudNum);
    allPixels(duds) = NaN;
    handle = pcolor(X,Y,allPixels);
    set(handle, 'EdgeColor', 'none');
    view(2);
    axis xy
else
    [xm,ym] = find(allPixels ~= dudNum);
    zm = ones(length(xm),1);
    for ix = 1:length(xm)
        val = allPixels(xm(ix),ym(ix));
        zm(ix)= val;
    end
    FDATATEST=scatteredInterpolant(ym, xm, zm, 'natural', 'none');
    dspl=1.0;  % use 1 for contours/quivers
    df0=1.0;
    [xq,yq]=meshgrid(xmin:df0:xmax, ymin:dspl:ymax);
    vq=FDATATEST(xq,yq);
    m = mesh(xq, yq, vq); %, vq);
    m.LineStyle = 'none';
    m.FaceColor = 'interp';
    hold on
    %con = contour(axH, xq, yq, vq, 'LineColor', 'k');
    %con = contour(axH, xq, yq, vq, conLevels, 'LineColor', 'k');
%     [px,py] = gradient(vq,100,100);
%     quiver(xq, yq, px, py, 2.0);

if bFlip > 0
    axis ij
    view(0, -90);
else
    view(0, 90);
end

    hold off
    %         for c=1:nClusters
    %         end
end

if (bOverlay > 0)
    for i = 1 : size(overArray, 1)
        y = overArray(i, 2) + round(offsetSPL) - 0.5;
        x = overArray(i, 1) - 0.5;
        rectangle('Position', [x y 1 1], 'FaceColor', 'none', 'EdgeColor', [0.3 0.3 0.3]);
    end
end

colormap(axH, colors);
caxis(axH, [cmin cmax]);
xlim(axH, [xmin xmax])
ylim(axH, [ymin ymax])
grid on


if bColorBar == 1
    cb = colorbar(axH);
%     cb.Location = 'north';
%     cb.Position(1) = axH.Position(1) + 0.01;
%     cb.Position(2) = axH.Position(2) + 0.1;
%     cb.Position(3) = cb.Position(3) / 2;
%     cb.Position(4) = cb.Position(4) / 2;
%     cb.TickLength = 0.05; 
%     cb.AxisLocation = 'out';
    if size(ticks) > 0
        cb.Ticks = ticks;
        cb.TickLabels = tickLabels;
    end
%     cb.Label.String = cbLabel; 
%     cb.Label.VerticalAlignment = 'bottom';
%     cb.Label.HorizontalAlignment = 'left';
%     cb.Label.Position(2) = cb.Position(2) - 0.4;
end
pbaspect([1.5 1 1]);
if plotHz == 1
    fMin = 220*2^((xmin-57)/12);
    fMax = 220*2^((xmax-57)/12); 
    if (fMax > 1600)
        step = 2;
    else 
        step = 1;
    end;
    ticks = [];
    tickLabels = {};
    ix = 1; 
    for j = 1 : step : 20
        i = (2^(j-1))*100;
        if (i >= fMin) & (i <= fMax)
            st = 57+12*log(i/220)/log(2);
            ticks(ix) = st;
            tickLabels(ix) = {num2str(i)};
            ix = ix + 1;
        end
    end
    axH.XTick = ticks;
    axH.XTickLabel = tickLabels;
end

end