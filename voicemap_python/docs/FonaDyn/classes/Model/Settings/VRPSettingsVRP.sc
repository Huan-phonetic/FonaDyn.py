// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
// Copyright (C) 2016-2023 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
VRPSettingsVRP {
	// States
	var <>isVisible;
    var <>loadedVRPdata;
	var <>vrpLoaded;
	var <>clarityThreshold;
	var <>bStacked;
	var <>bSingerMode;
	var <>bHzGrid;
	var <>wantsContextSave;

	*new {
		^super.new.init;
	}

	init {
		clarityThreshold = 0.96;
		bStacked = true;
		bSingerMode = false;
		bHzGrid = false;
		isVisible = true;
		wantsContextSave = false;
		vrpLoaded = false;
	}

	setLoadedData { arg newData;
		loadedVRPdata = newData;
		vrpLoaded = true;
	}

	getLoadedData {
		vrpLoaded = false;
		^loadedVRPdata
	}
}