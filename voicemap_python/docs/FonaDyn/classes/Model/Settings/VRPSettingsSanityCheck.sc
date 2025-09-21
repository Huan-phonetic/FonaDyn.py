// Copyright (C) 2016-2024 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
//
// Addition of a sanity check for all the settings.
// May help to contain all the sanityChecks in the same file,
//	 instead of small checks in each individual settings file.
//

+ VRPSettings {
	sanityCheck {
		var ret = true;

		ret = csdft.sanityCheck(this);
		ret = ret and: { io.sanityCheck(this) };
		^ret;
	}
}

+ VRPSettingsCSDFT {
	sanityCheck { | settings |
		var ret = true;
		var tmpHarm =
		[
			settings.cluster.nHarmonics,
			settings.sampen.amplitudeHarmonics,
			settings.sampen.phaseHarmonics
		].maxItem.asInteger;

		if (tmpHarm != nHarmonics, {
			format("Harmonics count mismatch: nHarmonics=%, needed=%", nHarmonics, tmpHarm).error;
			nHarmonics = tmpHarm;
			settings.cluster.nHarmonics = tmpHarm;
			ret = false;
		});

		^ret;
	}
}

+ VRPSettingsIO {
	sanityCheck { | settings |
		var retVal = true;
		var ios = settings.io;

		// Protest if the specified input file does not exist
		if (ios.inputType == VRPSettingsIO.inputTypeFile(), {
			if (File.exists(ios.filePathInput).not,
				{
					format("File % not found.", ios.filePathInput.tr($\\, $/)).error;
					retVal = false;

				})
			});

		// Protest if we are about to overwrite the source signal when re-recording
		if (ios.enabledWriteAudio
			and: { ios.inputType == VRPSettingsIO.inputTypeFile() }
			and: { ios.keepInputName }
			and: { PathName(ios.filePathInput).pathOnly.asString.tr($\\, $/).beginsWith(settings.general.output_directory) },
			{
				"Choose another output directory, or uncheck \"Keep input file name\" to avoid overwriting signals.".error;
				retVal = false;
			}
		);

		^retVal;
	}
}