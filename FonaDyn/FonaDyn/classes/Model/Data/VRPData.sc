// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
//
// General manager of the data to present in the views.

VRPData {
	var <io;
	var <csdft;
	var <cluster;
	var <clusterPhon;
	var <vrp;
	var <scope;
	var <player;
	var <general;

	var <settings; // A deep copy of the settings made on each start

	*new { | s |
		^super.new.init(s);
	}

	init { | s |
		settings = s.deepCopy;
		s.class.metrics[VRPSettings.iClarity].minVal = settings.vrp.clarityThreshold;

		io = VRPDataIO(settings);
		csdft = VRPDataCSDFT(settings);
		cluster = VRPDataCluster(settings);
		clusterPhon = VRPDataClusterPhon(settings);
		vrp = VRPDataVRP(settings);
		scope = VRPDataScope(settings);
		player = VRPDataPlayer(settings);
		general = VRPDataGeneral(settings);

	}

	// This method creates a big multiline string that can be saved as a script.
	// Running that script will restore the context that made the given map.
	mapContextString {  arg mapPathName;
		var lines = List.newClear(0);
		var lineStr, quotedPath, totalStr;

		lineStr = "//// FonaDyn version " ++ VRPMain.mVersion.asString ++ " context script ////";
		lines.add(lineStr);

		lineStr = Date.localtime.format("//// Created %H:%M:%S, %Y-%m-%d");
		lines.add(lineStr);

		lineStr = "general.output_directory=" ++ settings.general.output_directory.quote;
		lines.add(lineStr);

		lineStr = "io.filePathInput=" ++ settings.io.filePathInput.tr($\\, $/).quote;
		lines.add(lineStr);

		lineStr = "io.enabledWriteLog=" ++ settings.io.enabledWriteLog.asString;
		lines.add(lineStr);

		lineStr = "vrp.clarityThreshold=" ++ settings.vrp.clarityThreshold.asString;
		lines.add(lineStr);

		lineStr = "scope.noiseThreshold=" ++ settings.scope.noiseThreshold.asString;
		lines.add(lineStr);

		lineStr = "sampen.amplitudeWindowSize=" ++ settings.sampen.amplitudeWindowSize;
		lines.add(lineStr);

		lineStr = "sampen.amplitudeHarmonics=" ++ settings.sampen.amplitudeHarmonics;
		lines.add(lineStr);

		lineStr = "sampen.amplitudeSequenceLength=" ++ settings.sampen.amplitudeSequenceLength;
		lines.add(lineStr);

		lineStr = "sampen.amplitudeTolerance=" ++ settings.sampen.amplitudeTolerance;
		lines.add(lineStr);

		lineStr = "sampen.phaseWindowSize=" ++ settings.sampen.phaseWindowSize;
		lines.add(lineStr);

		lineStr = "sampen.phaseHarmonics=" ++ settings.sampen.phaseHarmonics;
		lines.add(lineStr);

		lineStr = "sampen.phaseSequenceLength=" ++ settings.sampen.phaseSequenceLength;
		lines.add(lineStr);

		lineStr = "sampen.phaseTolerance=" ++ settings.sampen.phaseTolerance;
		lines.add(lineStr);

		lineStr = "cluster.initialize=true";
		lines.add(lineStr);

		lineStr = "clusterPhon.initialize=true";
		lines.add(lineStr);

		lineStr = "cluster.learn=false";
		lines.add(lineStr);

		lineStr = "clusterPhon.learn=false";
		lines.add(lineStr);

		lineStr = "LOAD " ++ settings.cluster.filePath.tr($\\, $/).quote;
		lines.add(lineStr);

		lineStr = "LOAD " ++ settings.clusterPhon.filePath.tr($\\, $/).quote;
		lines.add(lineStr);

		quotedPath = mapPathName.tr($\\, $/).quote;
		lineStr = "LOAD " ++ quotedPath;
		lines.add(lineStr);

		// Ask VRPSettings to check the file-mod times
		// and alert user if the cluster data has changed.
		lineStr = format("checkClusterFileMods(%)", quotedPath);
		lines.add(lineStr);

		// Context is now complete
		lines.add(format("HOLD // Context restored for %", quotedPath));

		totalStr = "";
		lines.do ( { | str | totalStr = totalStr ++ str ++ "\n" } );
		^totalStr;
	}

}


