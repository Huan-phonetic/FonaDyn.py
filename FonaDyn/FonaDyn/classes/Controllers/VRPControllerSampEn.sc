// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
/**
 * Manages the buffers, synths and settings concerning the calculations of the Sample Entropy.
 */

// TODO: Fix the fetch

VRPControllerSampEn {
	var mData;
	var mTarget;
	var mSynth;

	*new { | target, data |
		^super.new.init(target, data);
	}

	// Init given the target and output data structure
	// This function is called once only!
	init { | target, data |
		mTarget = target;
		mData = data; // General data
	}

	// Tell the busManager what buses it requires.
	// This function is called before prepare.
	requires { | busManager |
		var d = mData;
/**		var sd = d.sampen;  **/
		var s = d.settings;
		var ss = s.sampen;

		busManager
		.requireAudio(\AmplitudeFirst, ss.amplitudeHarmonics )
		.requireAudio(\PhaseFirst, ss.phaseHarmonics )
		.requireAudio(\GateFilteredDFT)
		.requireAudio(\SampEn);
	}

	// Prepare to start - init the SynthDefs and allocate Buffers.
	// This function is always called before start.
	prepare { | libname, server, busManager, clock |
		var d = mData;
/**		var sd = d.sampen;	**/
		var s = d.settings;
		var ss = s.sampen;

		VRPSDSampEn.compile(
			libname,
			ss.amplitudeWindowSize,
			ss.amplitudeHarmonics,
			ss.amplitudeSequenceLength,
			ss.amplitudeTolerance,
			ss.phaseWindowSize,
			ss.phaseHarmonics,
			ss.phaseSequenceLength,
			ss.phaseTolerance
		);
	}

	// Start - Create the synths, and initiate fetching of data at the regular
	// interval given by the clock parameter.
	// This function is always called between prepare and stop.
	start { | server, busManager, clock |
		var bm = busManager;

		// Instantiate the Synths
		mSynth = Synth(*VRPSDSampEn.sampEn(
			bm.audio(\AmplitudeFirst),
			bm.audio(\PhaseFirst),
			bm.audio(\GateFilteredDFT),
			bm.audio(\SampEn),
			mTarget,
			\addToTail)
		);
	}

	// Free the synths and buffers after finishing fetching data.
	// The synths are guaranteed to be paused at this point - so buffers should be stable.
	stop {
		// Free the Synths
		mSynth.free;
	}

	sync { nil } // Nothing to do
}