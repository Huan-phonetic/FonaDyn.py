// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
//
// General manager of settings.
//
VRPSettings {
	var <io;
	var <sampen;
	var <csdft;
	var <>cluster;
	var <>clusterPhon;
	var <vrp;
	var <scope;
	var <player;
	var <general;
	var <waitingForStash;

	// classvar, because we need access from all over the place, but only one original instance
	classvar <metrics;			// in order sorted by ID
	classvar <metricsDict; 		// not ordered, access by symbol

	// Metric layer ID numbers
	// These numbers define the positions in the "Layers:" drop-down menu.
	const
	<iDensity = 0,			// must be 0
	<iClarity = 1,			// must be 1
	<iCrestFactor = 2,		// must be 2
	<iSpecBal = 3,
	<icppSmoothed = 4,
	<iEntropy = 5,
	<idEGGmax = 6,
	<iQcontact = 7,
	<iIcontact = 8,
	<iHRFEGG = 9,  <iLastMetric = 9,
	<iClustersEGG = 10,
	<iClustersPhon = 11;

	*new {
		^super.new.init;
	}

	waitingForStash_ { | bWaiting |
		waitingForStash = bWaiting;
	// postf("waitingForStash set to %\n", bWaiting);
	}

	init {
		var tmpMetrics;

		// Initialize the array 'metrics' and the Dictionary 'metricsDict'
		metricsDict = Dictionary();
		tmpMetrics = VRPMetric.subclasses collect: { | cm |	cm.new() } ;
		tmpMetrics do: { | cm | metricsDict.put(cm.class.symbol, cm) } ;
		// The order from .subclasses is random,
		// so put the metrics in the desired order
		metrics = Array.newClear(tmpMetrics.size);
		tmpMetrics do: { | m, i |
			var mIx = m.class.metricNumber;
			// MAYBE CHECK AGAINST OVERWRITING WITH MULTIPLE OCCURRENCES OF THE SAME metricNumber
			metrics[mIx] = m;
		};

		// Check if user has configured custom colors
		if ((~colorSetFileName.notNil) and: { ~colorSetFileName.isEmpty.not }, {
			this.installColorMappings(~colorSetFileName)
		});

		// Create the various *Settings objects
		io = VRPSettingsIO();
		sampen = VRPSettingsSampEn();
		csdft = VRPSettingsCSDFT();
		cluster = VRPSettingsCluster();
		clusterPhon = VRPSettingsClusterPhon();
		vrp = VRPSettingsVRP();
		scope = VRPSettingsScope();
		player = VRPSettingsPlayer();
		general = VRPSettingsGeneral();
		waitingForStash = false;
	}

	// Set up any optional metric color-mappings here
	installColorMappings { | csvPath |
		var fPath = csvPath;
		var fName = PathName(csvPath).fileName;

		if (fName == csvPath, {
			// No folder given: add the default file location
			csvPath = ~fdExtensions +/+ "FonaDynTools/colormaps" +/+ fName;
			csvPath = csvPath.tr($\\, $/);
		});

		block { |break|
			if (File.exists(csvPath).not, {
				format("Could not find file: %", csvPath).error;
				break.value(-1);
			});

			// Open the colorset file and check validity
			if (csvPath.toLower.endsWith("_colorset.csv"), {
				var cArray=nil;

				cArray = FileReader.read(
					csvPath,
					skipEmptyLines: true,
					skipBlanks: true,
					delimiter: VRPMain.cListSeparator
				);
				// If the first row contains only one element, it might be comma-delimited (not semicolon).
				// Try to parse it as such. This saves hassle when reading CSV files from elsewhere.
				if (cArray[0].size == 1, {
					cArray.clear;
					cArray = FileReader.read(csvPath, skipEmptyLines: true, skipBlanks: true, delimiter: $,);
				});

				// Row 0 is a heading row
				cArray[1..] do: { | row, ix |
					var mName = row[0];
					var mapFileName = row[1];
					var minColor = row[2];
					var maxColor = row[3];
					var minValue = row[4];
					var maxValue = row[5];
					if (mName.isEmpty.not, {
						metrics.do ({ | m, ix |
							if (m.csvName == mName, {
								var ret = m.installColorMapping(mapFileName, minColor, maxColor, minValue, maxValue);
								if (ret < 0, { break.value(ret) });
							});
						});
					});
				};
			});
		}
	}

	archive { | symArchive |
		var a = ();
		// We can't save the root .settings object
		// because it contains open functions
		general.saveSettingsOnExit = false;
		a[\general] = general;
		a[\io] 		= io;
		a[\sampen] 	= sampen;
		a[\csdft]   = csdft;
		a[\cluster] = cluster;
		a[\clusterPhon] = clusterPhon;
		// The loadedVRPdata are not really settings,
		// yet would take most of the space in the archive.
		vrp.loadedVRPdata = nil;
		a[\vrp] 		= vrp;
		a[\scope]		= scope;
		// settings.player is not needed

		Archive.global.put(symArchive, a);
	}

	unarchive { | symArchive |
		var u;

		u = Archive.global.at(symArchive);
		if (u.notNil, {
			general = u[\general];
			general.saveSettingsOnExit = false;
			general.guiChanged = true;
			io = u[\io];
			sampen = u[\sampen];
			csdft = u[\csdft];
			cluster = u[\cluster];
			clusterPhon = u[\clusterPhon];
			vrp = u[\vrp];
			scope = u[\scope];
			^true;
		},{
			format("NOTE: No .rerun settings have been archived for %", symArchive).postln;
			^false;
		});
		// Settings are saved in VRPMain: mWindow.onClose
	}

	edit { | assignment |
		var cmdStr, fnCmd;
		// Compiles and runs a line of code
		// that usually sets a member variable of 'this'
		cmdStr = "arg s; s." ++ assignment;
		fnCmd = cmdStr.compile;  // This works only in SC 3.13.0 and higher - reliably?
		fnCmd.(this);
	}

/*
	edit { | assignment |
		// Not pretty to use a global variable,
		// but I don't know how else to do it
		// var cmdStr = "this." ++ assignment; // out of scope in .interpret
		var cmdStr;
		if (assignment.isEmpty.not, {
			cmdStr = "~gVRPMain.mContext.model.settings." ++ assignment;
			interpret(cmdStr);
		});
	}
*/
	checkMapContext {
		var retval = true;
		if (io.filePathInput.isNil, { retval = false } );
		if (cluster.initialize.not, { retval = false } );
		if (cluster.learn,			{ retval = false } );
		if (cluster.filePath.isNil, { retval = false } );
		if (clusterPhon.learn, 		{ retval = false } );
		if (clusterPhon.initialize.not, { retval = false } );
		if (clusterPhon.filePath.isNil, { retval = false } );
		^retval
	}

	checkClusterFileMods { | mapPathName |
		var eggTime, phonTime, mapTime;
		var eggName, phonName, mapName;
		var retval = true;

		mapName = PathName(mapPathName).fileName;

		if (cluster.filePath.isNil,
			{ retval = false },
			{
				eggTime = File.mtime(cluster.filePath);
				eggName = PathName(cluster.filePath).fileName;
			}
		);

		if (clusterPhon.filePath.isNil,
			{ retval = false },
			{
				phonTime = File.mtime(clusterPhon.filePath);
				phonName = PathName(clusterPhon.filePath).fileName;
			}
		);

		if (retval, {
			mapTime = File.mtime(mapPathName) + 5;
			if ((mapTime < eggTime), {
				format("The file % was changed later than %", eggName, mapName).warn;
			});

			if ((mapTime < phonTime), {
				format("The file % was changed later than %", phonName, mapName).warn;
			});
		});
		^retval
	}
}