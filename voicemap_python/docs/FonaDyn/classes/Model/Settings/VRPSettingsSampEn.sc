// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
VRPSettingsSampEn {
	// States
	var <>isVisible;
	var <>amplitudeWindowSize; // The size of the window (in cycles) where we look for matching sequences via the Sample Entropy algorithm
	var <>amplitudeHarmonics; // The number of harmonics we use to produce the SampEn measurement
	var <>amplitudeSequenceLength; // The length of a matching sequence
	var <>amplitudeTolerance; // The tolerance for matching values in the sequences

	var <>phaseWindowSize; // The size of the window (in cycles) where we look for matching sequences via the Sample Entropy algorithm
	var <>phaseHarmonics; // The number of harmonics we use to produce the SampEn measurement
	var <>phaseSequenceLength; // The length of a matching sequence
	var <>phaseTolerance; // The tolerance for matching values in the sequences

	// States of check boxes - the VRPMenuSampEn has been promoted to a general graph plotter
	var <>bDrawSpecBal;
	var <>bDrawCPP;
	var <>bDrawSampEn;
	var <>bDrawQci;
	var <>bDrawDEGGmax;
	var <>bDrawIc;			// not used
	var <>bDrawCrest;		// not used

	var <>limit; // Limit for producing arrows

	*new {
		^super.new.init;
	}

	init {
		limit = 0.1;
		isVisible = true;
	}

	graphsRequested {
		var g = [];
		if (bDrawSpecBal, { g = g.add(\DelayedSpecBal)});
		if (bDrawCPP,     { g = g.add(\CPPsmoothed)});		// smoothed or not
		if (bDrawSampEn,  { g = g.add(\SampEn)});
		if (bDrawQci,     { g = g.add(\Qcontact)});
		if (bDrawDEGGmax, { g = g.add(\DEGGmax)});
//		if (bDrawIc,      { g = g.add(\Icontact)});
//		if (bDrawCrest,   { g = g.add(\Crest)});
		^g
	}

}