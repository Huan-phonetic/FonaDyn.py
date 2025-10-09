function [colors, minVal, maxVal] = FonaDynColors(strMetric, nClusters)
%%function [colors, minVal, maxVal] = FonaDynColors(strMetric, nClusters)
% This function creates a color array to match the colors in FonaDyn v2.1.
%  strMetric can be any of the names of columns > 2 in the _VRP.csv file
%  as saved from FonaDyn (not case sensitive), currently: 
%  'Total' | 'Clarity' | 'Crest' | 'SpecBal' | 'Entropy' | 'dEGGmax' | 'Qcontact' |
%  'Icontact' |'maxCluster' | 'Cluster <N>'   where <N> is a cluster number.
%
%  FonaDynColors returns an RGB color array for the Matlab "colormap" function,
%  as well as the minimum and maximum values for the Matlab "caxis" function.
%  The integer argument nClusters is used only with 'maxCluster' and 'Cluster <N>'
%  and is required but ignored in all other cases.

switch lower(strMetric)
    case 'total',
%  SuperCollider code from the FonaDyn *palette_ methods in "/FonaDyn/classes/Views/VRPColorMap.sc". 
% 	*paletteDensity {
% 		^{ | v |
% 			// map 1..<10000 to light...darker grey
% 			var cSat = v.clip(1, 10000).explin(1, 10000, 0.95, 0.25);
% 			Color.grey(cSat, 1);
% 		};
        intensity = (0.95:-0.01:0.25)';
        colors = [intensity intensity intensity]; 
        minVal = 0;  % here minVal and maxVal are the log10 
        maxVal = 4;  % of the actual values to be plotted

    case 'clarity',
% 	*paletteClarity { | threshold |
% 		VRPDataVRP.clarityThreshold_(threshold);  // register the new threshold
% 		^{ | v |
% 			// Map values above the threshold to a green shade (brighter green the better the clarity)
% 			// Map values below the threshold to gray
% 			if (v > VRPDataVRP.clarityThreshold,
% 				Color.green(v.linlin(VRPDataVRP.clarityThreshold, 1.0, 0.5, 1.0)),
% 				Color.gray)
% 		};
% 	}
        intensity = (0.49:0.01:1.0)';
        colors = [zeros(size(intensity)) intensity zeros(size(intensity))]; 
        minVal = 0.96;   % replace with current clarity threshold 
        maxVal = 1;

    case 'crest',
% 	*paletteCrest {
% 		^{ | v |
% 			var cHue;
% 
% 			// map crest factor 1.414 (+3 dB) ... <4 (+12 dB) to cyan...green...red
% 			cHue = v.clip(0, 5.0).explin(1.414, 4, 0.333, 0);
% 			Color.hsv(cHue, 1, 1);
% 		};
% 	}
% 
        hues = (0.33 : -0.01 : 0)';
        hsvs = [hues ones(size(hues)) ones(size(hues))];
        colors = hsv2rgb(hsvs);
        minVal = 1.414;     % crest factors here in linear
        maxVal = 4; 
        
    case 'specbal',
% 	*paletteSpecBal {
% 		^{ | v |
% 			var cHue;
% 
% 			// map spectrum balance -42... 0 (dB) to green...red
% 			cHue = v.linlin(-42.0, 0.0, 0.333, 0);
% 			Color.hsv(cHue, 1, 1);
% 		};
% 	}
        hues = (0.33 : -0.01 : 0)';
        hsvs = [hues ones(size(hues)) ones(size(hues))];
        colors = hsv2rgb(hsvs);
        minVal = -42;     % specbal in dB
        maxVal = 0; 
        
    case 'cpps',
% 	*paletteCPP {
% 		^{ | v |
% 			var cHue;
% 
% 			// map CPP smoothed 0...+20 (dB) to blue...red
% 			cHue = v.linlin(0.0, 20.0, 0.666, 0.0);
% 			Color.hsv(cHue, 1, 1);
% 		};
% 	}
% 
        hues = (0.666 : -0.02 : 0)';
        hsvs = [hues ones(size(hues)) ones(size(hues))];
        colors = hsv2rgb(hsvs);
        minVal = 0;     % CPPs here in dB
        maxVal = 20; 
        
    case 'entropy',
% *paletteEntropy {
%     ^{ | v |
%         var sat;
%         // Brown, saturated at 20. Should be scaled for nHarmonics in SampEn
%         sat = v.clip(1, 20).linlin(1, 20, 0.1, 1.0);
%         Color.white.blend(Color.new255(165, 42, 42), sat);
%     };
% }
        reds = (1.0 : (165-255)/(30*255) : 165/255)';  reds(1)=0.9;
        greblus = (1.0 : (42-255)/(30*255) : 42/255)'; greblus(1)=1;
        colors = [reds greblus greblus];
        minVal = 0;
        maxVal = 10;

    case 'deggmax'  % same as Qdelta
% 	*paletteDEGGmax {
% 		^{ | v |
% 			var cHue;
% 			// alias Qdelta, in publications
% 			// map log(dEGGmax) 1.0 ... <20 to green...red
% 			cHue = v.explin(1, 20, 0.333, 0);
% 			Color.hsv(cHue, 1, 1)
% 		};
% 	}
        hues = [(0.33 : -0.005 : 0)]';
        hsvs = [hues ones(size(hues)) ones(size(hues))];
        colors = hsv2rgb(hsvs);
        minVal = 0;     % log10 (dEGGnt)
        maxVal = 1.0;   % 1.3 (20) in FonaDyn, use 1.0 (10) for FD-based metrics
        
    case 'qcontact',
% 	*paletteQcontact {
% 		^{ | v |
% 			var cHue;
% 
% 			// map large Qc=0.6 to red, small Qc=0.1 to purple
% 			cHue = v.linlin(0.1, 0.6, 0.83, 0.0);
% 			Color.hsv(cHue, 1, 1)
% 		};
% 	}
        hues = (0.83 : -0.01 : 0)';
        hsvs = [hues ones(size(hues)) ones(size(hues))];
        colors = hsv2rgb(hsvs);
        minVal = 0.1;     % Qcontact linear mapping
        maxVal = 0.6; 
        
    case 'qspeed',
% 	*paletteQspeed {
%       THE SPEED QUOTIENT Qsi IS NOT IMPLEMENTED IN FONADYN 2.1
% 	}
        hues = [(0.33 : -0.005 : 0)]';
        hsvs = [hues ones(size(hues)) ones(size(hues))];
        colors = hsv2rgb(hsvs);
        minVal = 0.0;     % log10 (0.2)
        maxVal = 0.7;   % log10(5) 
                
    case 'icontact',
% 	*paletteIcontact {
% 		^{ | v |
% 			var cHue;
% 
% 			// map large Ic=1.0 to red, small Ic=0 to blue
% 			cHue = v.linlin(0.0, 0.7, 0.67, 0.0);
% 			Color.hsv(cHue, 1, 1)
% 		};
% 	}
        hues = [0.67 : -0.01 : 0]';
        hsvs = [hues ones(size(hues)) ones(size(hues))];
        colors = hsv2rgb(hsvs);
        minVal = 0;     % Icontact mapping
        maxVal = 0.6; 
        
    % Special: build a stack of (nClusters x 10) graded cluster colors    
    case 'maxcluster'
%         colors=zeros(10*nClusters, 3);
%         cmap = colormapFD(nClusters, 0.7);
%         for c=1:nClusters
%             for i=1:10
%                 hmap = rgb2hsv(cmap(c,:));
%                 hmap(2) = hmap(2)*(i/10.0);
%                 colors((c-1)*10+i,:) = hsv2rgb(hmap);
%             end
%         end       
%         minVal = 0;
%         maxVal = nClusters;

    %colorblindness friendly
        colors=zeros(10*nClusters, 3);
        cmap = getColorFriendly(nClusters);
        for c=1:nClusters
            for i=1:10
                hmap = rgb2hsv(cmap(c,:));
                hmap(2) = hmap(2)*(i/10.0);
                colors((c-1)*10+i,:) = hsv2rgb(hmap);
            end
        end       
        minVal = 0;
        maxVal = nClusters;
        
    otherwise
        nCluster = sscanf(lower(strMetric), 'cluster %u');
        if isnumeric(nCluster)
% *paletteCluster { | typeColor |
%     ^{ | v |
%         // Blend with white depending on the count. Counts >= 200 aren't blended at all.
%         var sat, cSat;
%         sat = v.clip(1, 200).explin(1, 200, 0.75, 0);
%         cSat = typeColor.blend(Color.white, sat);
%         cSat
%     };
% }
            cmap = colormapFD(nClusters, 1);
            hue = rgb2hsv(cmap(nCluster, :));
            sats = (0.1 : 0.02 : 0.7)';
            hsvs = [ones(size(sats))*hue(1) sats ones(size(sats))];
            colors = hsv2rgb(hsvs);
            minVal = 0;
            maxVal = log10(200);
end 
end

