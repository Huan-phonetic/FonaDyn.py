// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
/**
 * Manages the buffers, synths and settings concerning I/O.
 */

VRPControllerIO {
	var mData;
	var mTargetInput;
	var mTargetOutput;
	var mGroupMain;
	var mClock;
	var mCondition;

	// OSCFuncs
	var mOSCFuncEOF;
	var mOSCFuncClip;

	// Buffers
	var mBufferFile;
	var mBufferAudio;
	var mBufferCycleDetectionLog;
	// var mBufferWritePoints;
	// var mBufferWriteSampEn;
	var mBufferWriteFrequencyAmplitude;
	var mBufferWriteLog;
	var mBufferWriteGates;
	var mBufferRecordExtraChannels;
	var mBufferNoOfFileInputChannels;
	var mBufferInputSampleFormat;
    var mBufferInputScaler;

	// Synths
	var mSynthInput;
	var mSynthEcho;
	var mSynthWriteAudio;
	var mSynthWriteCycleDetection;
	// var mSynthWritePoints;
	// var mSynthWriteSampEn;
	// var mSynthWriteFrequencyAmplitude;
	var mSynthWriteLog;
	var mSynthWriteGates;
	var mSynthRecordExtraChannels;

	// Constants
	classvar triggerIDEOF = 117;
	classvar triggerIDClip = 118;

	// Static Input/Output Buses
	classvar <audioInputBusIndexVoice = 0; // First audio input channel
	classvar <audioInputBusIndexEGG = 1; // Second audio input channel
	classvar audioOutputBusIndexSpeaker = 0; // First audio output channel

	// Settings
	classvar mBitDepthStr = "int24"; 	// New default setting from v2.6.6
	classvar <logBaseTracks = 13;		// Adding phoncluster made 13

	// If this duration is too short, the file might be closed before all writes have completed
	// - resulting in a useless file, or potentially an erronous file.
	// The duration in seconds to wait for all disk writes to finish.
	classvar diskWriteWaitTime = 2.0;

	*new { | targetInput, targetOutput, mainGroup, data |
		^super.new.init(targetInput, targetOutput, mainGroup, data);
	}

	*configureInputs { arg inputVoice, inputEGG;
		if (inputVoice.notNil, { audioInputBusIndexVoice = inputVoice });
		if (inputEGG.notNil, { audioInputBusIndexEGG = inputEGG });
	}

	*configureSampleFormat { arg sampleFormat;
		if (sampleFormat.notNil,  {
			mBitDepthStr = if (sampleFormat == "int24", "int24", "int16");
		});
	}

	// Init given the target and output data structure
	// This function is called once only!
	init { | targetInput, targetOutput, mainGroup, data |
		mTargetInput = targetInput;
		mTargetOutput = targetOutput;
		mGroupMain = mainGroup;
		mBufferNoOfFileInputChannels = 2;
		mBufferInputScaler = 1.0;
		mData = data; // General data
		mCondition = Condition(true); // Syncing has no effect until we free the buffers.
	}

	// Tell the busManager what buses it requires.
	// This function is called before prepare.
	requires { | busManager |
		var d = mData;
		var s = d.settings;
		var m = s.clusterPhon.nMetrics;
		var n = s.cluster.nHarmonics;
		var h = s.csdft.nHarmonics;

		busManager
		.requireAudio(\EchoVoice)
		.requireAudio(\EchoEGG)
		.requireAudio(\ConditionedMicrophone)
		.requireAudio(\ConditionedEGG)
		.requireAudio(\GateCycle)
		.requireAudio(\GateDelayedCycle)
		.requireAudio(\GateFilteredDFT)
		.requireAudio(\DeltaAmplitudeFirst, n)
		.requireAudio(\DeltaPhaseFirst, n)
		.requireAudio(\SampEn)
		.requireAudio(\DEGGmax)
		.requireAudio(\Qcontact)
		.requireAudio(\Icontact)
		.requireAudio(\ClusterNumber)
		.requireControl(\PhonClusterNumber)
		.requireAudio(\Timestamp)
		.requireAudio(\DelayedFrequency)
		.requireAudio(\DelayedAmplitude)
		.requireAudio(\DelayedClarity)
		.requireAudio(\DelayedCrest)
		.requireAudio(\DelayedSpecBal)
		.requireControl(\CPPsmoothed)
		.requireControl(\NoiseThreshold)
		.requireAudio(\AmplitudeFirst, h)
		.requireAudio(\PhaseFirst, h);
	}

	// Prepare to start - init the SynthDefs and allocate Buffers.
	// This function is always called before start.
	prepare { | libname, server, busManager, clock |
		var d = mData;
		var iod = d.io;
		var s = d.settings;
		var ios = s.io;
		var cs = s.cluster;
		var cds = s.csdft;

		var timestamp = d.general.timestamp;
		var dir = s.general.output_directory;
		var base_path = dir +/+ timestamp;

		var oldname = (ios.filePathInput ? "").basename;
		var ix = (oldname.findBackwards("_Voice_EGG.wav", true) ? -1 ) - 1;

		if (ios.keepInputName and: (ix >= 0), // and: (ios.enabledWriteAudio.not)),
			{ base_path = dir +/+ (oldname[0..ix])}
		);

		if (File.type(dir) != \directory, { File.mkdir(dir); }); // Create the directory if necessary.

		// Find out how many channels the input file has
		if (ios.inputType == VRPSettingsIO.inputTypeFile(), {
			var nChannels;
			var sf = SoundFile.new;
			sf.openRead(ios.filePathInput);
			mBufferNoOfFileInputChannels = sf.numChannels;
			// If the input file contains floats, assume that they are calibrated in Pascals
			if (sf.sampleFormat == "float", { mBufferInputScaler = 0.07071 });
			sf.close;
		});

		VRPControllerIO.configureInputs(
			inputVoice: ios.arrayRecordInputs[0],
			inputEGG:   ios.arrayRecordInputs[1]
		);

		// Compile the synths for IO
		VRPSDIO.compile(libname, triggerIDEOF, triggerIDClip, mGroupMain.nodeID,
			cds.nHarmonics, ios.enabledEGGlisten, ios.writeLogFrameRate,
			ios.arrayRecordExtraInputs, ios.rateExtraInputs, mBufferNoOfFileInputChannels, mBufferInputScaler,
			ios.arrayRecordInputs, busManager.control(\NoiseThreshold)
		);

		// Allocate buffers if necessary
		if (ios.inputType == VRPSettingsIO.inputTypeFile(), {
			mBufferFile = Buffer.cueSoundFile(server, ios.filePathInput, 0, mBufferNoOfFileInputChannels, 131072);
		});

		if (ios.enabledWriteAudio, {
			// If the SPL range is wide then override the bits setting
			if (VRPDataVRP.nMaxSPL == 140.0, { mBitDepthStr = "int24" });

			iod.filePathAudio = base_path ++ "_Voice_EGG.wav";
			File.open(iod.filePathAudio, "w").close; // Forcibly create the file

			mBufferAudio = Buffer.alloc(server, 131072, ios.arrayRecordInputs.size);  // was (..., 2);
			mBufferAudio.write(iod.filePathAudio, "wav", mBitDepthStr, 0, 0, true);
		});

		if (ios.enabledWriteCycleDetection, {
			iod.filePathCycleDetectionLog = base_path ++ "_CycleDetection.wav";
			File.open(iod.filePathCycleDetectionLog, "w").close; // Forcibly create the file
			mBufferCycleDetectionLog = Buffer.alloc(server, 131072, 2);
			mBufferCycleDetectionLog.write(iod.filePathCycleDetectionLog, "wav", "int16", 0, 0, true);
		});

/*		if (ios.enabledWriteOutputPoints, {
			iod.filePathOutputPoints = base_path ++ "_Points.aiff";
			File.open(iod.filePathOutputPoints, "w").close; // Forcibly create the file
			mBufferWritePoints = Buffer.alloc(server, cs.nDimensions * 1000, cs.nDimensions);
			mBufferWritePoints.write(iod.filePathOutputPoints, "aiff", "float", 0, 0, true);
		});

		if (ios.enabledWriteSampEn, {
			iod.filePathSampEn = base_path ++ "_SampEn.aiff";
			File.open(iod.filePathSampEn, "w").close; // Forcibly create the file
			mBufferWriteSampEn = Buffer.alloc(server, server.sampleRate, 1);
			mBufferWriteSampEn.write(iod.filePathSampEn, "aiff", "float", 0, 0, true);
		});

		if (ios.enabledWriteOutputPoints or: ios.enabledWriteSampEn, {
			iod.filePathFrequencyAmplitude = base_path ++ "_FreqAmp.aiff";
			File.open(iod.filePathFrequencyAmplitude, "w").close; // Forcibly create the file
			mBufferWriteFrequencyAmplitude = Buffer.alloc(server, server.sampleRate, 2);
			mBufferWriteFrequencyAmplitude.write(iod.filePathFrequencyAmplitude, "aiff", "float", 0, 0, true);
		});
*/
		if ( ios.enabledWriteLog, {
			var channels = logBaseTracks + (2 * (cds.nHarmonics+1));
			iod.filePathLog = base_path ++ "_Log.aiff";
			File.open(iod.filePathLog, "w").close; // Forcibly create the file
			mBufferWriteLog = Buffer.alloc(server, channels * 400, channels);
			mBufferWriteLog.write(iod.filePathLog, "aiff", "float", 0, 0, true);
		});

		if ( ios.enabledWriteGates and: (ios.writeLogFrameRate == 0),
			{
				iod.filePathGates = base_path ++ "_Gates.wav";
				File.open(iod.filePathGates, "w").close; // Forcibly create the file
				mBufferWriteGates = Buffer.alloc(server, 131072, 5);
				mBufferWriteGates.write(iod.filePathGates, "wav", "int16", 0, 0, true);
		});

		if (ios.enabledWriteAudio and: ios.enabledRecordExtraChannels, {
			var bitsStr = "int16";
			iod.filePathExtraChannels = base_path ++ "_Extra.wav";
			File.open(iod.filePathExtraChannels, "w").close; // Forcibly create the file
			mBufferRecordExtraChannels = Buffer.alloc(server, server.sampleRate, ios.arrayRecordExtraInputs.size);
			mBufferRecordExtraChannels.sampleRate = ios.rateExtraInputs;
			if (ios.rateExtraInputs == 44100, { bitsStr = mBitDepthStr });
			mBufferRecordExtraChannels.write(iod.filePathExtraChannels, "wav", bitsStr, 0, 0, true);
		});
	}

	// Start - Create the synths, and initiate fetching of data at the regular
	// interval given by the clock parameter.
	// This function is always called between prepare and stop.
	start { | server, busManager, clock |
		var bm = busManager;
		var d = mData;
		var iod = d.io;
		var s = d.settings;
		var ios = s.io;
		mClock = clock;

		// Instantiate the input synth
		switch (ios.inputType,
			VRPSettingsIO.inputTypeFile(), {
				// mSynthInput = Synth(*VRPSDIO.diskInput(
				// mSynthInput = Synth(*VRPSDIO.diskInput2(
				// 	mBufferFile,
				mSynthInput = Synth(*VRPSDIO.diskInput3(
					mBufferFile,
					bm.control(\NoiseThreshold),
					bm.audio(\EchoVoice),
					bm.audio(\EchoEGG),
					bm.audio(\ConditionedMicrophone),
					bm.audio(\ConditionedEGG),
					mTargetInput,
					\addToTail)
				);

				// Setup the OSCFunc
				mOSCFuncEOF =
				OSCFunc({ iod.eof = true; },
					'/tr',
					server.addr,
					nil,
					[mSynthInput.nodeID, triggerIDEOF]
				);
			},

			VRPSettingsIO.inputTypeRecord(), {
/*				if (s.scope.noiseThreshold == 0.0,
					{	// No noise reduction
					mSynthInput = Synth(*VRPSDIO.liveInput(
					audioInputBusIndexVoice,
					audioInputBusIndexEGG,
					bm.audio(\EchoVoice),
					bm.audio(\EchoEGG),
					bm.audio(\ConditionedMicrophone),
					bm.audio(\ConditionedEGG),
					mTargetInput,
					\addToTail)
					);
				}, {
*/
				// Noise reduction, but also delay
				mSynthInput = Synth(*VRPSDIO.liveInput2(
					audioInputBusIndexVoice,
					audioInputBusIndexEGG,
					bm.control(\NoiseThreshold),
					bm.audio(\EchoVoice),
					bm.audio(\EchoEGG),
					bm.audio(\ConditionedMicrophone),
					bm.audio(\ConditionedEGG),
					mTargetInput,
					\addToTail)
				);

				// Setup the OSCFunc
				mOSCFuncClip =
				OSCFunc({ iod.clip = true; },
					'/tr',
					server.addr,
					nil,
					[mSynthInput.nodeID, triggerIDClip]
				);
			}
		);

		if (ios.enabledEcho, {
			switch (ios.inputType,
				VRPSettingsIO.inputTypeRecord(), {
					var selectEGG = if (s.scope.noiseThreshold > 0.0, \ConditionedEGG, \EchoEGG);
					mSynthEcho = Synth(*VRPSDIO.echoMicrophone(
						0.0,
						bm.audio(\EchoVoice),
						// bm.audio(\EchoEGG),
						bm.audio(selectEGG),
						audioOutputBusIndexSpeaker,
						mTargetOutput,
						\addToTail)
				)},
				VRPSettingsIO.inputTypeFile(), {
					mSynthEcho = Synth(*VRPSDIO.echoMicrophone(
						0.15,  // Delay the sound a further 150 ms, to sync with the visuals
						bm.audio(\ConditionedMicrophone),
						bm.audio(\ConditionedEGG),
						audioOutputBusIndexSpeaker,
						mTargetOutput,
						\addToTail)
				)}
			);
		});

		if (ios.enabledWriteAudio, {
			switch (ios.inputType,
				VRPSettingsIO.inputTypeRecord(), {
					mSynthWriteAudio = Synth(*VRPSDIO.writeAudio(
						bm.audio(\EchoVoice),
						bm.audio(\EchoEGG),
						mBufferAudio,
						mTargetOutput,
						\addToTail)
				)},
				VRPSettingsIO.inputTypeFile(), {
					mSynthWriteAudio = Synth(*VRPSDIO.writeAudio(
						bm.audio(\ConditionedMicrophone),
						bm.audio(\ConditionedEGG),
						mBufferAudio,
						mTargetOutput,
						\addToTail)
			)});
		});

		if (ios.enabledWriteCycleDetection, {
			mSynthWriteCycleDetection = Synth(*VRPSDIO.writeCycleDetectionLog(
				bm.audio(\ConditionedEGG),
				bm.audio(\GateCycle),
				mBufferCycleDetectionLog,
				mTargetOutput,
				\addToTail)
			);
		});

/*		if (ios.enabledWriteOutputPoints, {
			mSynthWritePoints = Synth(*VRPSDIO.writePoints(
				bm.audio(\GateFilteredDFT),
				bm.audio(\DeltaAmplitudeFirst),
				bm.audio(\DeltaPhaseFirst),
				mBufferWritePoints,
				mTargetOutput,
				\addToTail)
			);
		});

		if (ios.enabledWriteSampEn, {
			mSynthWriteSampEn = Synth(*VRPSDIO.writeSampEn(
				bm.audio(\GateFilteredDFT),
				bm.audio(\SampEn),
				mBufferWriteSampEn,
				mTargetOutput,
				\addToTail)
			);
		});

		if (ios.enabledWriteOutputPoints or: ios.enabledWriteSampEn, {
			mSynthWriteFrequencyAmplitude = Synth(*VRPSDIO.writeFrequencyAmplitude(
				bm.audio(\GateFilteredDFT),
				bm.audio(\DelayedFrequency),
				bm.audio(\DelayedAmplitude),
				mBufferWriteFrequencyAmplitude,
				mTargetOutput,
				\addToTail)
			);
		});
*/
		if ( ios.enabledWriteLog, {
			mSynthWriteLog = Synth(*VRPSDIO.writeLog(
				bm.audio(\GateFilteredDFT),
				bm.audio(\Timestamp),
				bm.audio(\DelayedFrequency),
				bm.audio(\DelayedAmplitude),
				bm.audio(\DelayedClarity),
				bm.audio(\DelayedCrest),
				bm.audio(\DelayedSpecBal),
				bm.control(\CPPsmoothed),
				bm.audio(\ClusterNumber),
				bm.control(\PhonClusterNumber),
				bm.audio(\SampEn),
				bm.audio(\Icontact),
				bm.audio(\DEGGmax),
				bm.audio(\Qcontact),
				bm.audio(\AmplitudeFirst),
				bm.audio(\PhaseFirst),
				mBufferWriteLog,
				mTargetOutput,
				\addToTail)
			);
		});

		if ( ios.enabledWriteGates and: ios.enabledWriteLog, {
			// (ios.enabledWriteLog or: ios.enabledWriteOutputPoints or: ios.enabledWriteSampEn), {
				mSynthWriteGates = Synth(*VRPSDIO.writeGates(
					bm.audio(\EchoEGG),
					bm.audio(\ConditionedEGG),
					bm.audio(\GateCycle),
					bm.audio(\GateDelayedCycle),
					bm.audio(\GateFilteredDFT),
					mBufferWriteGates,
					mTargetOutput,
					\addToTail)
				);
		});

		if (ios.enabledWriteAudio and: ios.enabledRecordExtraChannels, {
			mSynthRecordExtraChannels = Synth(*VRPSDIO.recordExtraChannels(
				mBufferRecordExtraChannels,
				mTargetOutput,
				\addToTail)
			);
		});

	}

	// Free the synths and buffers after finishing fetching data.
	// The synths are guaranteed to be paused at this point - so buffers should be stable.
	stop {
		// Free synths
		[
			mSynthInput,
			mSynthEcho,
			mSynthWriteAudio,
			mSynthWriteCycleDetection,
			// mSynthWritePoints,
			// mSynthWriteSampEn,
			// mSynthWriteFrequencyAmplitude,
			mSynthWriteLog,
			mSynthWriteGates,
			mSynthRecordExtraChannels
		]
		do: { | synth | if (synth.notNil, { synth.free; }); };

		// ...and buffers
		mCondition.test = false; // Enable waiting on the condition variable.

		mClock.sched( mClock.tempo * diskWriteWaitTime, {
			[
				mBufferFile,
				mBufferAudio,
				mBufferCycleDetectionLog,
				// mBufferWritePoints,
				// mBufferWriteSampEn,
				// mBufferWriteFrequencyAmplitude,
				mBufferWriteLog,
				mBufferWriteGates,
				mBufferRecordExtraChannels
			]
			do: { | buf, ix |
				if (buf.notNil, {
					mCondition.test = false;
					buf.close ({ | b |
						b.free;
						mCondition.test = true;
						mCondition.signal;
					});
					mCondition.wait;
				});
			};

			mCondition.test = true; // Ensure that no-one get stuck on wait after we exit here
			mCondition.signal; // Signal anyone who's stuck waiting
		});

		// ...and OSCFuncs
		if (mOSCFuncClip.notNil, {mOSCFuncClip.free});
		if (mOSCFuncEOF.notNil, {mOSCFuncEOF.free});
	}

	// Put the actual sample rate into the _Extra.wav file header
	patchWAVheader {
		var sfFast, sfSlow, sampleCount, data, extraPath;
		extraPath = mData.io.filePathExtraChannels;

		// We can't modify only the file metadata with SClang,
		// so we must load the entire Extra file into memory,
		// which is why the sampling rate should not be very high
		sfFast = SoundFile.new;
		sfFast.openRead(extraPath);
		sampleCount = sfFast.numFrames * sfFast.numChannels;
		data = FloatArray.newClear(sampleCount);
		sfFast.readData(data);
		sfFast.close;

		// ... then DELETE the disk file - otherwise the old header takes precedence...(!)
		// The .delete seems to take a little time on some systems, so we keep trying
		"Patching ".post;
		while ({ File.exists(extraPath) }, {
			File.delete(extraPath);
			".".post;
			0.1.wait;
		});
		" done!".postln;

		// ...finally, write back the data with the actual sample rate
		sfSlow = SoundFile.new(extraPath);
		sfSlow.numChannels = sfFast.numChannels;
		sfSlow.headerFormat = "wav";
		sfSlow.sampleFormat = "int16";
		sfSlow.sampleRate = mData.settings.io.rateExtraInputs;
		sfSlow.openWrite;
		sfSlow.writeData(data);
		sfSlow.close;
		data.free;
		mCondition.test = true; // Ensure that no-one get stuck on wait after we exit here
		mCondition.signal; // Signal anyone who's stuck waiting
	}

	sync {
		// Wait for all writes to finish
		mCondition.wait;
		// Patch the Extra.wav file, if applicable
		if (mData.settings.io.enabledWriteAudio and: mData.settings.io.enabledRecordExtraChannels, {
			if (mData.settings.io.rateExtraInputs <= 500, {
				mCondition.test = false; // Enable waiting on the condition variable.
				fork {
					this.patchWAVheader;
				};
				mCondition.wait;  // wait for patching to finish
			});
			format("Extra channels % written to %",
				mData.settings.io.arrayRecordExtraInputs.asString,
				mData.io.filePathExtraChannels)
			.postln;
		});
	}
}
