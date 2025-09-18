// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
VRPSettingsDialog {
	var mDialog, mView;
	var mStaticTextProgramVersion;
	var mStaticTextProgramLicence;
	var mStaticTextCycleSeparationMethod;
	var mButtonCycleSeparationMethod;
	var mStaticTextClarityThreshold;
	var mNumberBoxClarityThreshold;
	var mCheckBoxPlayEGG;
	var mCheckBoxEnableHighSPL;
	var mCheckBoxKeepInputName;
	var mCheckBoxSaveMapContext;
	var mCheckBoxWriteGates;
	var mCheckBoxShowDiagnostics;
	var mCheckBoxSuppressGibbs;
	var mCheckBoxSaveSettingsOnExit;
	var mStaticTextChannels;
	var mEditTextChannels;
	var mStaticTextExtraChannels;
	var mEditTextExtraChannels;
	var mMenuExtraRates;
	var mStaticTextExtraRate, mStaticTextExtraRate2;
	var mStaticTextColorTheme;
	var mListColorThemes;

	var mButtonOK, mButtonCancel;

	*new { | parentMenu |
		^super.new.init(parentMenu);
	}

	init { | parentMenu |
		var static_font = Font.new(\Arial, 9, usePointSize: true);
		var extraRates = #[50, 60, 100, 150, 210, 252, 300, 350, 401, 450, 490, 44100];

		mDialog = Window.new("FonaDyn settings" /*, resizable: false */);
		mView = mDialog.view;
		mView.background_( Color.grey(0.65) );
		mView.font_(static_font);

		mStaticTextProgramVersion
		= StaticText.new(mView, Rect())
		.string_("Program version:" + VRPMain.mVersion.asString);

		mStaticTextProgramLicence
		= StaticText.new(mView, Rect())
		.string_("Distributed under EUPL v1.2.\n(click to read)" )
		.align_(\right)
		.stringColor_(Color.blue);
		mStaticTextProgramLicence
		.fixedSize_(mStaticTextProgramLicence.sizeHint)
		.mouseDownAction_( { ~gLicenceLink.openOS } );  // global link defined in VRPMain.sc

		mStaticTextCycleSeparationMethod
		= StaticText.new(mView, Rect())
		.string_("Cycle separation method:")
		.align_(\right);

		mButtonCycleSeparationMethod
		= Button(mView, Rect())
		.states_([["Phase tracker Φ"], ["Peak follower Λ"]])
		.value_(parentMenu.oldSettings.csdft.method);

		mStaticTextClarityThreshold
		= StaticText.new(mView, Rect())
		.string_("Clarity threshold:")
		.align_(\right);

		mNumberBoxClarityThreshold
		= NumberBox(mView, Rect())
		.clipLo_(0.5)
		.clipHi_(1.0)
		.step_(0.01)
		.scroll_step_(0.01)
		.value_(parentMenu.oldSettings.vrp.clarityThreshold);

		mCheckBoxPlayEGG
		= CheckBox(mView, Rect(), "Play the EGG signal on the second output")
		.value_(parentMenu.oldSettings.io.enabledEGGlisten);

		mCheckBoxKeepInputName
		= CheckBox(mView, Rect(), "Keep input file name up to _Voice_EGG.wav")
		.value_(parentMenu.oldSettings.io.keepInputName);

		mCheckBoxSaveMapContext
		= CheckBox(mView, Rect(), "Save Map (if green) also saves a context script")
		.value_(parentMenu.oldSettings.vrp.wantsContextSave);

/*		mCheckBoxEnableHighSPL
		= CheckBox(mView, Rect(), "Set max SPL to 140 dB - 'singer mode'")
		.value_(parentMenu.oldSettings.vrp.bSingerMode);
*/
		mCheckBoxShowDiagnostics
		= CheckBox(mView, Rect(), "Show additional diagnostic features")
		.value_(parentMenu.oldSettings.general.enabledDiagnostics);

		mCheckBoxWriteGates
		= CheckBox(mView, Rect(), "Write _Gates file with any cycle-synchronous output")
		.value_(parentMenu.oldSettings.io.enabledWriteGates);

		mCheckBoxSuppressGibbs
		= CheckBox(mView, Rect(), "Suppress Gibbs' ringing in resynthesized EGG shapes")
		.value_(parentMenu.oldSettings.cluster.suppressGibbs);

		mStaticTextChannels
		= StaticText.new(mView, Rect())
		.string_("Record inputs:")
		.fixedWidth_(120)
		.align_(\right);

		mEditTextChannels
		= TextField.new(mView, Rect())
		.string_( parentMenu.oldSettings.io.arrayRecordInputs.asString )
		.fixedWidth_(142)
		.align_(\left);

		mStaticTextExtraChannels
		= StaticText.new(mView, Rect())
		.string_("Record extra inputs:")
		.fixedWidth_(120)
		.align_(\right);

		mEditTextExtraChannels
		= TextField.new(mView, Rect())
		.string_(if (parentMenu.oldSettings.io.enabledRecordExtraChannels,
			{ parentMenu.oldSettings.io.arrayRecordExtraInputs.asString},
			{ nil.asString }))
		.fixedWidth_(142)
		.align_(\left);

		mMenuExtraRates = PopUpMenu(mView, Rect(10, 0, 40, 23))
		.items_(extraRates.collect({|v| v.asString}));
		mMenuExtraRates.valueAction_(extraRates.indexOf(parentMenu.oldSettings.io.rateExtraInputs));

		mStaticTextExtraRate = StaticText.new(mView, Rect())
		.string_("at")
		.align_(\right);

		mStaticTextExtraRate2 = StaticText.new(mView, Rect())
		.string_("Hz")
		.align_(\left);

		mStaticTextColorTheme
		= StaticText.new(mView, Rect())
		.string_("Window dressing:")
		.fixedWidth_(120)
		.align_(\topRight);

		mListColorThemes = ListView(mView, Rect(0, 0, 40, 0))
		.fixedHeight_(64)
		.items_([ "Grand Piano", "Night Flight", "Nordic Deco", "Army Surplus" ])
		.font_(static_font)
		.selectionMode_(\single)
		.fixedWidth_(142)
		.value_(parentMenu.oldSettings.general.colorThemeKey);

		mCheckBoxSaveSettingsOnExit
		= CheckBox(mView, Rect(), "Save all settings, for FonaDyn.rerun")
		.value_(parentMenu.oldSettings.general.saveSettingsOnExit);

		mButtonCancel
		= Button(mView, Rect())
		.states_([["Cancel"]])
		.action_({ mDialog.close });

		mButtonOK
		= Button(mView, Rect())
		.states_([["OK"]])
		.action_( { this.accept(parentMenu) });

		mView.allChildren( { |v|
			v.font_(static_font);
			if (v.class == StaticText, { v.fixedWidth_(160); v.fixedHeight_(35) });
		});

		mStaticTextProgramVersion
		.font_(static_font.boldVariant)
		.fixedWidth_(180);

		mView.keyDownAction_({ | view, char |
			case
			{char == 27.asAscii} { mDialog.close; true }			// Escape: Cancel
			{char == 13.asAscii} { this.accept(parentMenu); true }	// Enter:  OK
			{ false }
		});

		mView.layout = VLayout.new(
			HLayout([mStaticTextProgramVersion, stretch: 1, align: \topLeft], [nil, s:3],
				[mStaticTextProgramLicence, stretch: 1, align: \right]),
			[nil, s:5],
			HLayout([mStaticTextCycleSeparationMethod, s: 1], [mButtonCycleSeparationMethod, s: 1], [nil, s: 1]),
			HLayout([mStaticTextClarityThreshold, s: 1], [mNumberBoxClarityThreshold, s: 1], [nil, s: 1]),
			[mCheckBoxPlayEGG, s: 1],
			// [mCheckBoxEnableHighSPL, s: 1],
			[mCheckBoxKeepInputName, s: 1],
			[mCheckBoxSaveMapContext, s: 1],
			[mCheckBoxShowDiagnostics, s: 1],
			[mCheckBoxWriteGates, s: 1],
			[mCheckBoxSuppressGibbs, s: 1],
			HLayout([mStaticTextChannels, s: 0], [mEditTextChannels, s: 0], nil),
			HLayout([mStaticTextExtraChannels, s: 0],
					[mEditTextExtraChannels, s: 0],
				    [mStaticTextExtraRate, s: 0],
					[mMenuExtraRates, s: 0],
					[mStaticTextExtraRate2, s: 0], nil),
			HLayout([mStaticTextColorTheme, s: 0], [mListColorThemes, s: 0], nil),
			[25, s:1],
			HLayout([mCheckBoxSaveSettingsOnExit, s: 1, a: \left], [nil, s: 20], [mButtonCancel, a: \right], [mButtonOK, a: \right])
		);

		// Don't know why this does not work until down here
		mStaticTextProgramVersion.font_(Font.new(\Arial, 9, true, false, true));
		mStaticTextProgramVersion.fixedSize_(mStaticTextProgramVersion.sizeHint);

		mView.layout.margins_(5);
		mDialog.front;
	} /* init */

	accept { | parentMenu |
		var arrAudio, arrExtra, bCheckChannels, bExtra;
 		parentMenu.newSettings.vrp.clarityThreshold_(mNumberBoxClarityThreshold.value);
		parentMenu.newSettings.csdft.method =
		switch( mButtonCycleSeparationMethod.value,
			0, VRPSettingsCSDFT.methodPhasePortrait,
			1, VRPSettingsCSDFT.methodPeakFollower
		);
		parentMenu.newSettings.io.enabledEGGlisten_(mCheckBoxPlayEGG.value);
		// parentMenu.newSettings.vrp.bSingerMode_(mCheckBoxEnableHighSPL.value);
		parentMenu.newSettings.io.enabledWriteGates_(mCheckBoxWriteGates.value);
		parentMenu.newSettings.general.enabledDiagnostics_(mCheckBoxShowDiagnostics.value);
		parentMenu.newSettings.io.keepInputName_(mCheckBoxKeepInputName.value);
		parentMenu.newSettings.vrp.wantsContextSave_(mCheckBoxSaveMapContext.value);

		// Parse the array of inputs to be recorded
		bCheckChannels = value {
			arrAudio = mEditTextChannels.string.compile.value;
			if (arrAudio.isNil or: arrAudio.isKindOf(Array).not,
				{ false },
				{ arrAudio.every({arg item, i; item.isKindOf(Number)}) }
			)
		};
		parentMenu.newSettings.io.arrayRecordInputs = if (bCheckChannels, { arrAudio }, [0, 1] );

		// Settings for any extra channels
		bExtra = value {
			arrExtra = mEditTextExtraChannels.string.compile.value;
			if (arrExtra.isNil or: arrExtra.isKindOf(Array).not,
				{ false },
				{ arrExtra.every({arg item, i; item.isKindOf(Number)}) }
			)
		};
		parentMenu.newSettings.io.enabledRecordExtraChannels = bExtra;
		parentMenu.newSettings.io.arrayRecordExtraInputs = if (bExtra, { arrExtra }, { nil } );
		parentMenu.newSettings.io.rateExtraInputs = mMenuExtraRates.item.asInteger;

		parentMenu.newSettings.cluster.suppressGibbs = mCheckBoxSuppressGibbs.value;
		parentMenu.newSettings.general.colorThemeKey_(mListColorThemes.value);
		parentMenu.newSettings.general.saveSettingsOnExit_(mCheckBoxSaveSettingsOnExit.value);
		parentMenu.bSettingsChanged_(true);
		mDialog.close;
	}

}