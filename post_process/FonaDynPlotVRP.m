function mSymbol = FonaDynPlotVRP(vrpArray, colNames, colName, fig, varargin)
%% function FonaDynPlotVRP(vrpArray, colNames, colName, fig, ...)
% <vrpArray> is an array of numbers and <colNames> is a cell array of column names, 
% both previously returned by FonaDynLoadVRP.m. 
% <colName> is the name of the column to be plotted (case sensitive).
% <fig> is the number of the current figure or subplot.
% Optional arguments: 
% 'MinCycles', integer       - set a minimum cycles-per-cell threshold
% 'Range', [foMin, foMax, Lmin, Lmax]   - specify plot range, in MIDI and dB
% 'OffsetSPL', value         - offset the dB SPL (for calibration)
% 'ColorBar', 'on' or 'off'  - show a horizontal color bar at the top left
% 'PlotHz', 'on' or 'off'    - plot frequency axis in Hz rather than in MIDI
% 'Mesh', 'on' or 'off'      - plot an interpolated mesh rather than cells
% 'Region', <overArray>      - plot rectangles over all cells in <overArray>
%                            - <overArray is in the same format as vrpArray>

minCycles = 1;
xmax = 91;  % can be <=96
xmin = 32;  % can be >=30
ymax = 120; % can be <=120
ymin = 30;  % can be >=40
offsetSPL = 0;  % useful for SPL re-calibration
bColorBar = 0;
bMesh = 0;
SDscale = 0.0;
plotHz = 0;
cbLabel = '';
tickLabels = {}; 
ticks = [];
bOverlay = 0;
bFlip = 1;
bSpecial = 0;

args = nargin-4;

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
        case 'SDscale'
            SDscale = varargin{i+1};
        case 'Region'
            overArray = varargin{i+1};
            if size(overArray, 1) > 0
               bOverlay = 1;
            end
        case 'Special'
            bSpecial = varargin{i+1};
        otherwise
            warning (['Unrecognized option: ' varargin{i}]);
    end
end

colPlot = find(ismember(colNames, colName)); 
if size(colPlot,2) < 1
    warning(['Unrecognized column name: ' colName]);
    return
end
colMaxCluster = find(ismember(colNames, 'maxCluster')); 
colTotal = find(ismember(colNames, 'Total')); 
% nClusters = size (vrpArray, 2) - colMaxCluster;
% add a colomn so nClusters should be:
nClusters = size (vrpArray, 2) - colMaxCluster;

nCluster = -1;

if SDscale <= 0.0
    % Set up a colormap to give similar colors as in FonaDyn
    [colors, cmin, cmax] = FonaDynColors(colName, nClusters);
else
    cmin = 0.0;
    cmax = SDscale;
    intensity = (0.95:-0.05:0.2)';
    fullW = ones(size(intensity,1),1)*0.95;
    % Make magenta, light to dark
    colors = [fullW intensity fullW]; 
end

allPixels = ones(150,100)*NaN;

indices = find((vrpArray(:, colTotal) >= minCycles ) & (vrpArray(:,1) >= xmin));
for i=1:length(indices)
    y = vrpArray(indices(i), 2) + round(offsetSPL);
    x = vrpArray(indices(i), 1);
    z = vrpArray(indices(i), colPlot);
    allPixels(y, x) = z;
end
ym = vrpArray(indices, 2) + round(offsetSPL);
xm = vrpArray(indices, 1);
zm = vrpArray(indices, colPlot);
        
switch lower(colName)
    case 'total'
        allPixels = log10(allPixels);
        cbLabel = 'log_{10}(cycles)';
        tickLabels = {'0', '1', '2', '3', '4'};
        ticks = [0, 1, 2, 3, 4];
        conLevels = ticks;
    case 'clarity'
        cbLabel = 'Clarity (latest)';
    case 'crest' % scale it to dB (1..4) -> (0..12)
        if SDscale > 0.0
            cbLabel = 'SD crest factor';
            conLevels = [1:0.1:SDscale];
        else
            cbLabel = 'Mean crest factor';
            conLevels = [1:1:5];
        end    
    case 'cpps' % scale it to dB (0..+20) 
        if SDscale > 0.0
            cbLabel = 'CPPs Cell-SDs';
            conLevels = [1:0.1:SDscale];
        else
            tickLabels = {'0', '5', '10', '15', '20'};
            ticks = [0, 5, 10, 15, 20];
            cbLabel = 'CPPs Cell-Means';
            conLevels = [0:5:20];
        end 
    case 'specbal' % scale it to dB (-42..-6) 
        if SDscale > 0.0
            cbLabel = 'SD of SB';
            conLevels = [1:0.1:SDscale];
        else
            tickLabels = {'-40', '-30', '-20', '-10'};
            ticks = [-40, -30, -20, -10];
            cbLabel = 'Mean SB';
            conLevels = [-40:5:10];
            bFlip = 0; % SB values are negative, so a mesh appears on top
        end    
    case 'entropy'
        %subIx = find(allPixels == 0);
        %allPixels(subIx) = NaN;
            tickLabels = {'0', '5', '10', '15', '20'};
            ticks = [0:5:20];
        cbLabel = 'Mean SampEn'; %'Max SampEn';
        conLevels = [0:5:20];
    case 'deggmax'
        if SDscale > 0.0
            cbLabel = 'SD \itQ\rm_{\Delta}';
            conLevels = [1:1:SDscale];
        else
            negIx = find(allPixels <= 0);
            allPixels(negIx) = 0.01;   % patch out rare negative values 
            allPixels = log10(allPixels); 
            
            negIx = find(zm <= 0);
            zm(negIx) = 0.01;   % patch out rare negative values 
            zm = log10(zm);  
            
            tickLabels = {'1', '2', '4', '7', '10'};
            ticks = [0, 0.301, 0.602, 0.8451, 1];
            conLevels = ticks;
            cbLabel = 'Mean \itQ\rm_{\Delta}';
        end   
        if bSpecial > 0
            cbLabel = [cbLabel ' FD'];
        end
    case 'qcontact'
        if SDscale > 0.0
            cbLabel = 'SD \itQ\rm_{ci}';
        else
            % ticks = (0.1:0.1:0.6);
            cbLabel = 'Mean \itQ\rm_{ci}';
            conLevels = (0.0:0.1:1.0);
        end
        if bSpecial > 0
            cbLabel = [cbLabel ' FD'];
        end
    case 'qspeed' % WAS USED FOR AN ARTICLE BUT IS NOT IN v2.1
        if SDscale > 0.0
            cbLabel = 'SD \itQ\rm_{si}';
            conLevels = [1:1:SDscale];
        else
            allPixels = log10(allPixels); 
            zm = log10(zm);  
            tickLabels = {'1', '2', '3', '4', '5'};
            ticks = [0, 0.301, 0.4771, 0.602, 0.7];
            cbLabel = 'Mean \itQ\rm_{si}';
            conLevels = ticks;
        end
        if bSpecial > 0
            cbLabel = [cbLabel ' FD'];
        end
    case 'icontact'
        if SDscale > 0.0
            cbLabel = 'SD \itI\rm_c';
        else
            ticks = (0.0:0.2:0.6);
            conLevels = (0.0:0.1:1.0);
            tickLabels  = {'0', '0.2', '0.4', '0.6' };
            cbLabel = 'Mean \itI\rm_c';
        end
        if bSpecial > 0
            cbLabel = [cbLabel ' FD'];
        end
    case 'maxcluster'
    % Plot overlay of all clusters (max cycles on top)
        allPixels = ones(150,100)*NaN;
        for c = 1 : nClusters
            subIx = find((vrpArray(:, colMaxCluster)==c) & vrpArray(:,colTotal)>=minCycles);
            rel = vrpArray(subIx, colMaxCluster+c) ./ vrpArray(subIx, colTotal);
            for i=1:length(subIx)
                y = vrpArray(subIx(i),2) + round(offsetSPL);
                x = vrpArray(subIx(i),1);
                z = 0.999 * rel(i);
                allPixels(y, x) = c - 1 + z;
            end
            ticks(c) = c-0.5;
            tickLabels{c} = num2str(c);
        end
        cbLabel = 'Cluster';
        nCluster = 0;
        
    case 'clustering'
        nCluster = 5;% 5 k
        if nCluster >= 1
            subIx = find(allPixels == 0);
            allPixels(subIx) = NaN;
            subIx = find(allPixels > 0);
            allPixels(subIx) = allPixels(subIx);
            cbLabel = [colName];
            tickLabels = {'1', '2', '3', '4', '5'};
            ticks = [1.4, 2.2, 3, 3.8, 4.6 ];
        end
        
    otherwise     % one of the clusters
        nCluster = colPlot - colMaxCluster;
        if nCluster >= 1
            subIx = find(allPixels == 0);
            allPixels(subIx) = NaN;
            subIx = find(allPixels > 0);
            allPixels(subIx) = log10(allPixels(subIx));
            cbLabel = [colName ': Cycles'];
            tickLabels = {'1', '10', '100'};
            ticks = [0, 1, 2];
        end
end

mSymbol = cbLabel;

% Plot the cells astride the grid, not next to it
X = (1:100)-0.5;
Y = (1:150)-0.5;
% figure(fig);
axH = gca;
grid on

if (bMesh == 0) || (nCluster >= 0)
    handle = pcolor(X,Y,allPixels);
    set(handle, 'EdgeColor', 'none');
    view(2);
    axis xy
elseif (bMesh == 1)
    %[ym,xm,vm] = find(allPixels*1.0); 
    FDATATEST=scatteredInterpolant(xm, ym, zm, 'natural', 'none');
    dspl=1.0;  % use 1 for contours/quivers
    df0=1.0;
    [xq,yq]=meshgrid(xmin:df0:xmax, ymin:dspl:ymax);
    vq=FDATATEST(xq,yq);
    m = mesh(xq, yq, vq); %, vq);
    m.LineStyle = 'none';
    m.FaceColor = 'interp';
    hold on
    %con = contour(axH, xq, yq, vq, 'LineColor', 'k');
    con = contour(axH, xq, yq, vq, conLevels, 'LineColor', 'k');
%     [px,py] = gradient(vq,100,100);
%     quiver(xq, yq, px, py, 2.0);

if bFlip > 0
    axis ij
    view(0, -90);
else
    view(0, 90);
end
%end
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
xlim(axH, [xmin xmax]);
ylim(axH, [ymin ymax]);

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
    cb.Label.VerticalAlignment = 'bottom';
    %cb.Label.HorizontalAlignment = 'left';
    %cb.Label.Position(2) = cb.Position(2) - 0.4;
end

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