// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //

Dolansky {
	*ar { arg in, decay, coeff;
		var peakPlus  = FOS.ar(PeakFollower.ar(in.max(0), decay), coeff, coeff.neg, coeff);
		var peakMinus = FOS.ar(PeakFollower.ar(in.neg.max(0), decay), coeff, coeff.neg, coeff);
		^Trig1.ar(SetResetFF.ar(peakPlus, peakMinus), 0);
	}
}

// This pseudo-UGen for the crest factor is cycle-synchronous
// and more correct than the built-in Crest UGen.
CrestCycles {
	*ar { arg in;
		var rms, trig, out;

		trig = Dolansky.ar(in, 0.999, 0.99);
		rms = AverageOut.ar(in.squared, trig).sqrt;
		out = RunningMax.ar(in.abs, Delay1.ar(trig))/rms;
		^[rms, Latch.ar(out, trig)]
	}
}

SpectrumBalance {
	*ar { arg in;
		var levelLo, levelHi;
		levelLo = RMS.ar(BLowPass4.ar(in, 1500, 2), 50).ampdb;
		levelHi = RMS.ar( BHiPass4.ar(in, 2000, 2), 50).ampdb;
		^Sanitize.ar(levelHi-levelLo, -50.0);
	}
}



VRPSDVRP {
	classvar nameAnalyzeAudio = \sdAnalyzeAudio;
	classvar nameCPPsmoothed = \sdCPPsmoothed;

	*compile { | libname, bSmoothCPP=true |

		///////////////////////////////////////////////////////////////////////////////////////////////////////
		// Analyze Audio SynthDef
		///////////////////////////////////////////////////////////////////////////////////////////////////////  ;
		SynthDef(nameAnalyzeAudio,
			{ | aiBusConditionedMic,
				coBusFrequency,
				coBusAmplitude,
				coBusClarity,
				coBusCrest,
				coBusSB |

				var in, inDelayed, inrms, amp, freq, freqOut, crest, gate, clarity, specBal;

				in = In.ar(aiBusConditionedMic);

				// Delay the audio metrics so as to align approximately
				// with the extracted freq and clarity
				// 20 ms ≈ 23 - 3 (half an FFT frame, and approx one cycle and filter group delays)
				inDelayed = DelayN.ar(in, 0.02, 0.02);
				#inrms, crest = CrestCycles.ar(inDelayed);
				specBal = SpectrumBalance.ar(inDelayed);

				// The following line serves only to guard against true-zero audio in test files
				amp = Select.kr(InRange.kr(inrms, -1.0, 0.0), [inrms.ampdb, DC.kr(-100)]);

				// Integrator brings down the HF - but is that needed?
				# freq, clarity = Tartini.kr(Integrator.ar(in, 0.995), n: 2048, k: 0, overlap: 1024);
				freq = Sanitize.kr(freq.cpsmidi, 20);
				freqOut = VarLag.kr(freq, 0.02322, 0, \lin);  // new in v3.3.0

				Out.kr(coBusFrequency, [freqOut]);
				Out.kr(coBusAmplitude, [amp]);
				Out.kr(coBusClarity, [clarity]);
				Out.kr(coBusCrest, [crest]);
				Out.kr(coBusSB, [specBal]);
			}
		).add(libname);

		/////////////////////////////////////////////////////////////////////////////////////////////////////
		// Cepstral Peak Prominence SynthDef
		/////////////////////////////////////////////////////////////////////////////////////////////////////

		SynthDef.new( nameCPPsmoothed,
			{ | aiBusConditionedMic,
				coBusCPPsmoothed |

				var in, inWithDither, chain, cepsChain, fftBuffer, cepsBuffer;
				var cpp, cppSan, slope, intercept, maxcpp, maxix;
				var lowBin = 25, highBin = 367;  // 880 Hz down to 60 Hz
				var ditherAmp = 1000000.reciprocal;		// was 24000.reciprocal until v3.0.6d

				fftBuffer = LocalBuf(2048);		// Tried halving the bufsize but not good
				cepsBuffer = LocalBuf(1024);

				in = In.ar(aiBusConditionedMic, 1);			// Get the audio signal
				inWithDither = WhiteNoise.ar(ditherAmp, add: in);// Prevent divide-by-zero issues later in the chain
				chain = FFT(fftBuffer, inWithDither, wintype: 1);  // Hanning window
				cepsChain = Cepstrum(cepsBuffer, chain);	// Both buffers are now in polar form (mag,phase)
				if (bSmoothCPP, {
					cepsChain = PV_MagSmooth(cepsChain, 0.3);	// Approx 16 Hz LP1 filter
					cepsChain = PV_MagSmear(cepsChain, 3);		// Implements 7-bin smearing (+/- 3 bins mean)
				});

				// PeakProminence is a custom UGen that is provided with FonaDyn
				// Only the cpp result is used here
				#cpp, slope, intercept, maxcpp, maxix = PeakProminence.kr(cepsChain, lowBin, highBin);
				cppSan = Sanitize.kr(cpp, -1.0);
				cppSan = VarLag.kr(cppSan, 0.02322, 0, \lin);		// new in 3.3.0
				Out.kr(coBusCPPsmoothed, [cppSan]);
			}
		).add(libname);

	} // .compile

	*analyzeAudio { | aiBusConditionedMic, coBusFrequency, coBusAmplitude, coBusClarity, coBusCrest, coBusSB ...args |
		^Array.with(nameAnalyzeAudio,
			[
				\aiBusConditionedMic, aiBusConditionedMic,
				\coBusFrequency, coBusFrequency,
				\coBusAmplitude, coBusAmplitude,
				\coBusClarity, coBusClarity,
				\coBusCrest, coBusCrest,
				\coBusSB, coBusSB
			],
			*args
		);
	}

	*cppSmoothed { | aiBusConditionedMic, coBusCPPsmoothed ...args |
		^Array.with(nameCPPsmoothed,
			[
				\aiBusConditionedMic, aiBusConditionedMic,
				\coBusCPPsmoothed, coBusCPPsmoothed
			],
			*args
		);
	}

}