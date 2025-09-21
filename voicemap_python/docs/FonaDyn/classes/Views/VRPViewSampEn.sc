// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //

// This class was originally for a dedicated SampEn plotter.
// It has been promoted to a more general plotter, but the name has stuck.

VRPViewSampEn {
	// Views
	var mView;
	var mScopeViewer;

	// Controls
	var mStaticTextSampEn;
	var mStaticTextAmplitude;
	var mStaticTextPhase;
	var mStaticTextTolerance;
	var mStaticTextWindowSize;
	var mStaticTextSequenceLength;
	var mStaticTextHarmonics;

	var mNumberBoxToleranceAmplitude;
	var mNumberBoxWindowSizeAmplitude;
	var mNumberBoxSequenceLengthAmplitude;
	var mNumberBoxHarmonicsAmplitude;

	var mNumberBoxTolerancePhase;
	var mNumberBoxWindowSizePhase;
	var mNumberBoxSequenceLengthPhase;
	var mNumberBoxHarmonicsPhase;

	var mStaticTextDrawChoices;
	var mCheckBoxDrawQci;
	var mCheckBoxDrawDEGGmax;
	var mCheckBoxDrawSampEn;
	var mCheckBoxDrawCPP;
//	var mCheckBoxDrawIc;
//	var mCheckBoxDrawCrest;
	var mCheckBoxDrawSpecBal;
	var mCurveColors;

	var mCPPsymbol;

	var mbMouseDown, mXstart, mXnow, mXscale, mbStretchOK;
	var mTempDuration, mNewDuration;

	// Constants
	classvar nMinSampleEntropy = -0.01; // Minimum sample entropy point written
	classvar boxWidth = 32;
	classvar boxHeight = 20;

	// Settings

	*new { | view |
		^super.new.init(view);
	}

	init { | view |
		var gl1, gl2, vGap;

		mView = view;
		mNewDuration = 2.0;  // default duration of time plots
		mXscale = 1.0;
		// mCPPsymbol = VRPSettings.metrics[VRPSettings.icppSmoothed].class.symbol;
		mCPPsymbol = VRPDataVRP.cppStr;

		this.initMenu;

		// Create the scope viewer
		value {
			var max1 =
			mNumberBoxHarmonicsAmplitude.value.asInteger *
			SampleEntropyFromBus.upperBounds(
				mNumberBoxWindowSizeAmplitude.value.asInteger,
				mNumberBoxSequenceLengthAmplitude.value.asInteger
			);

			var max2 =
			mNumberBoxHarmonicsPhase.value.asInteger *
			SampleEntropyFromBus.upperBounds(
				mNumberBoxWindowSizePhase.value.asInteger,
				mNumberBoxSequenceLengthPhase.value.asInteger
			);

			var maxSampEn = max1 + max2;

			mScopeViewer = ScopeViewer(mView,
				ControlSpec(-1, 0, units: "s"),
				ControlSpec(nMinSampleEntropy, maxSampEn, units: "SampEn")
			);
		};
		mCurveColors = (0..4) collect: { |v| Color.hsv(v/6, 0.5, 1) };
		mScopeViewer.background_(Color.black);

		///// Implement mouse-drag control of the plot duration  ///////////
		mbMouseDown = false;
		mbStretchOK = true;

		mScopeViewer.viewGrid.mouseDownAction = { arg view, x, y, modifiers, buttonNumber, clickCount;
			if (mbStretchOK, {
				mXstart = view.bounds.width - x;
				mbMouseDown = true;
			});
		};

		mScopeViewer.viewGrid.mouseMoveAction = { arg view, x, y, modifiers;
			if (mbMouseDown, {
				mXnow   = (view.bounds.width - x).clip(0, view.bounds.width);
				mXscale = mXstart.asFloat / mXnow.asFloat;
				this.updateGraphOnly;
			});
		};

		mScopeViewer.viewGrid.mouseUpAction = { arg view, x, y, modifiers, buttonNumber, clickCount;
			if (mbStretchOK, {
				mbMouseDown = false;
				mXscale = 1.0;
				mXstart = mXnow;
				mNewDuration = mTempDuration;
			});
		};


		// Fill the check boxes with colours
		[
			mCheckBoxDrawSpecBal,
			mCheckBoxDrawCPP,
			mCheckBoxDrawSampEn,
			mCheckBoxDrawQci,
			mCheckBoxDrawDEGGmax
//			mCheckBoxDrawIc,
//			mCheckBoxDrawCrest,
		] do: { |c, i|	c.palette = c.palette.base_(mCurveColors[(i-3).mod(5)]) };

		// Force a grid redraw when any SampEn-related parameter is changed
		mView.allChildren do: { |v| if (v.class == NumberBox, { v.addAction( { mScopeViewer.refresh } ) }) } ;

		gl1 = GridLayout.columns(
			[
				mStaticTextDrawChoices,
				mCheckBoxDrawSpecBal,
				mCheckBoxDrawCPP,
				mCheckBoxDrawSampEn,
				mCheckBoxDrawQci,
				mCheckBoxDrawDEGGmax
//				mCheckBoxDrawIc
//				mCheckBoxDrawCrest
			]
		);
		vGap = mCheckBoxDrawSampEn.bounds.height/5;
		gl1.vSpacing_(vGap);

		gl2 = GridLayout.rows(
			[
				mStaticTextSampEn,
				mStaticTextAmplitude,
				mStaticTextPhase
			],
			[
				mStaticTextTolerance,
				mNumberBoxToleranceAmplitude,
				mNumberBoxTolerancePhase
			],
			[
				mStaticTextWindowSize,
				mNumberBoxWindowSizeAmplitude,
				mNumberBoxWindowSizePhase
			],
			[
				mStaticTextSequenceLength,
				mNumberBoxSequenceLengthAmplitude,
				mNumberBoxSequenceLengthPhase
			],
			[
				mStaticTextHarmonics,
				mNumberBoxHarmonicsAmplitude,
				mNumberBoxHarmonicsPhase
			]
		);
		gl2.setAlignment(mStaticTextAmplitude, \center);
		gl2.setAlignment(mStaticTextPhase, \center);
		4 do: { |i| gl2.setAlignment(0@(i+1), \right) };
		gl2.vSpacing_(vGap);

		mView.layout_(
			VLayout(
				[
					HLayout(
						[ gl1, stretch: 1 ],
						[ nil, stretch: 10],
						[ gl2, stretch: 1 ]
					)
				],
				[ mScopeViewer.view, stretch: 10]
			)
		);
		mView.layout.spacing_(1);
		mView.layout.margins_(5);
	}

	initMenu {
		var static_font = Font(\Arial, 8, usePointSize: true);
		var general_font = Font(\Arial, 9, italic: true, usePointSize: true);

		mStaticTextDrawChoices = StaticText(mView, Rect())
		.string_("      Plots")
		.font_(general_font);
		mStaticTextDrawChoices
		.fixedWidth_(mStaticTextDrawChoices.sizeHint.width)
		.stringColor_(Color.white);
		boxHeight = mStaticTextDrawChoices.sizeHint.height;

/*		mCheckBoxDrawIc = CheckBox(mView, Rect(), "Icontact (Ic) ×10")
		.font_(static_font)
		.value_(false);

		mCheckBoxDrawCrest = CheckBox(mView, Rect(), "Audio Crest Factor")
		.font_(static_font)
		.value_(false);   */

		mCheckBoxDrawSpecBal = CheckBox(mView, Rect(), "SpecBal (SB)(dB+40)/4")
		.font_(static_font)
		.value_(false);

		mCheckBoxDrawCPP = CheckBox(mView, Rect(), mCPPsymbol.asString ++ " (dB)")
		.font_(static_font)
		.value_(false);

		mCheckBoxDrawSampEn = CheckBox(mView, Rect(), "Cycle-rate SampEn →")
		.font_(static_font)
		.value_(false)
		.action_({ |b| this.sampEnControlsVisible(b.value) });

		mCheckBoxDrawQci = CheckBox(mView, Rect(), "Qcontact (Qci) ×10")
		.font_(static_font)
		.value_(false);

		mCheckBoxDrawDEGGmax = CheckBox(mView, Rect(), "Peak dEGG (QΔ)")
		.font_(static_font)
		.value_(false);

		mStaticTextSampEn = StaticText(mView, Rect())
		.string_("SampEn")
		.font_(general_font);
		mStaticTextSampEn
		.fixedWidth_(mStaticTextSampEn.sizeHint.width)
		.stringColor_(Color.white);

		mStaticTextAmplitude = StaticText(mView, Rect())
		.string_(" L ")
		.font_(general_font);
		mStaticTextAmplitude
		.fixedWidth_(mStaticTextAmplitude.sizeHint.width)
		.stringColor_(Color.white);

		mStaticTextPhase = StaticText(mView, Rect())
		.string_(" φ ")
		.font_(general_font);
		mStaticTextPhase
		.fixedWidth_(mStaticTextPhase.sizeHint.width)
		.stringColor_(Color.white);
		boxWidth = mStaticTextPhase.sizeHint.width*1.5;

		////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////

		mStaticTextTolerance = StaticText(mView, Rect())
		.string_("Tolerance ")
		.font_(static_font);
		mStaticTextTolerance
		.fixedWidth_(mStaticTextTolerance.sizeHint.width)
		.stringColor_(Color.white)
		.align_(\center);

		mNumberBoxToleranceAmplitude = NumberBox(mView, Rect())
		.value_(0.2)
		.clipLo_(0)
		.step_(0.1)
		.scroll_step_(0.1);

		mNumberBoxTolerancePhase = NumberBox(mView, Rect())
		.value_(0.4)
		.clipLo_(0)
		.step_(0.1)
		.scroll_step_(0.1);

		////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////

		mStaticTextWindowSize = StaticText(mView, Rect())
		.string_("Window ")
		.font_(static_font);
		mStaticTextWindowSize
		.fixedWidth_(mStaticTextWindowSize.sizeHint.width)
		.stringColor_(Color.white)
		.align_(\center);

		mNumberBoxWindowSizeAmplitude = NumberBox(mView, Rect())
		.value_(10)
		.clipLo_(2)
		.step_(1)
		.scroll_step_(1)
		.action_ { | nb |
			// Note that the sequence length cannot be larger than or equal to the window size.
			if (mNumberBoxSequenceLengthAmplitude.value > (nb.value - 1), {
				mNumberBoxSequenceLengthAmplitude.valueAction_( nb.value - 1 );
			});
			mNumberBoxSequenceLengthAmplitude.clipHi_(nb.value - 1);
		};

		mNumberBoxWindowSizePhase = NumberBox(mView, Rect())
		.value_(10)
		.clipLo_(2)
		.step_(1)
		.scroll_step_(1)
		.action_ { | nb |
			// Note that the sequence length cannot be larger than or equal to the window size.
			if (mNumberBoxSequenceLengthPhase.value > (nb.value - 1), {
				mNumberBoxSequenceLengthPhase.valueAction_( nb.value - 1 );
			});
			mNumberBoxSequenceLengthPhase.clipHi_(nb.value - 1);
		};

		////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////

		mStaticTextSequenceLength = StaticText(mView, Rect())
		.string_("Length ")
		.font_(static_font);
		mStaticTextSequenceLength
		.fixedWidth_(mStaticTextSequenceLength.sizeHint.width)
		.stringColor_(Color.white)
		.align_(\center);

		mNumberBoxSequenceLengthAmplitude = NumberBox(mView, Rect())
		.value_(1)
		.clipLo_(1)
		.clipHi_(mNumberBoxWindowSizeAmplitude.value - 1)
		.step_(1)
		.scroll_step_(1);

		mNumberBoxSequenceLengthPhase = NumberBox(mView, Rect())
		.value_(1)
		.clipLo_(1)
		.clipHi_(mNumberBoxWindowSizePhase.value - 1)
		.step_(1)
		.scroll_step_(1);

		////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////

		mStaticTextHarmonics = StaticText(mView, Rect())
		.string_("Harmonics ")
		.font_(static_font);
		mStaticTextHarmonics
		.fixedWidth_(mStaticTextHarmonics.sizeHint.width)
		.stringColor_(Color.white);

		mNumberBoxHarmonicsAmplitude = NumberBox(mView, Rect())
		.value_(4)
		.clipLo_(1)
		.clipHi_(20) // We don't know how many are actually available.
		.step_(1)
		.scroll_step_(1);

		mNumberBoxHarmonicsPhase = NumberBox(mView, Rect())
		.value_(4)
		.clipLo_(1)
		.clipHi_(20) // We don't know how many are actually available.
		.step_(1)
		.scroll_step_(1);

		mView.allChildren do: { | c, i |
			if (c.isKindOf(NumberBox), {
				c.font_(static_font);
				c.fixedWidth_(boxWidth);
				c.fixedHeight_(boxHeight);
			});
			if (c.isKindOf(StaticText), { c.fixedHeight_(boxHeight) });
			if (c.isKindOf(CheckBox),   { c.fixedHeight_(boxHeight) });
		};

		this.sampEnControlsVisible(false);
	} /* initMenu */

	enableInterface { | enable |
		[
			mCheckBoxDrawSampEn,
			mCheckBoxDrawQci,
			mCheckBoxDrawDEGGmax,
			mCheckBoxDrawCPP,
//			mCheckBoxDrawIc,
//			mCheckBoxDrawCrest,
			mCheckBoxDrawSpecBal,
			mNumberBoxToleranceAmplitude,
			mNumberBoxWindowSizeAmplitude,
			mNumberBoxSequenceLengthAmplitude,
			mNumberBoxHarmonicsAmplitude,
			mNumberBoxTolerancePhase,
			mNumberBoxWindowSizePhase,
			mNumberBoxSequenceLengthPhase,
			mNumberBoxHarmonicsPhase
		]
		do: { | ctrl | ctrl.enabled_(enable) };
		// mbStretchOK = enable;
	}

	sampEnControlsVisible { | b |
		[
			mStaticTextSampEn,
			mStaticTextAmplitude,
			mStaticTextPhase,
			mStaticTextTolerance,
			mStaticTextWindowSize,
			mStaticTextSequenceLength,
			mStaticTextHarmonics,
			mNumberBoxToleranceAmplitude,
			mNumberBoxWindowSizeAmplitude,
			mNumberBoxSequenceLengthAmplitude,
			mNumberBoxHarmonicsAmplitude,
			mNumberBoxTolerancePhase,
			mNumberBoxWindowSizePhase,
			mNumberBoxSequenceLengthPhase,
			mNumberBoxHarmonicsPhase
		]
		do: { | ctrl | ctrl.visible_(b) };
	}

	layout { ^mView; }

	stash { | settings |
		var ss = settings.sampen;

		mNumberBoxToleranceAmplitude.value_(ss.amplitudeTolerance);
		mNumberBoxWindowSizeAmplitude.value_(ss.amplitudeWindowSize);
		mNumberBoxSequenceLengthAmplitude.value_(ss.amplitudeSequenceLength);
		mNumberBoxHarmonicsAmplitude.value_(ss.amplitudeHarmonics);

		mNumberBoxTolerancePhase.value_(ss.phaseTolerance);
		mNumberBoxWindowSizePhase.value_(ss.phaseWindowSize);
		mNumberBoxSequenceLengthPhase.value_(ss.phaseSequenceLength);
		mNumberBoxHarmonicsPhase.value_(ss.phaseHarmonics);

		mCheckBoxDrawSpecBal.value_(ss.bDrawSpecBal);
		mCheckBoxDrawCPP.value_(ss.bDrawCPP);
		mCheckBoxDrawSampEn.valueAction_(ss.bDrawSampEn);
		mCheckBoxDrawQci.value_(ss.bDrawQci);
		mCheckBoxDrawDEGGmax.value_(ss.bDrawDEGGmax);
//		mCheckBoxDrawIc.value_(ss.bDrawIc);
//		mCheckBoxDrawCrest.value_(ss.bDrawCrest);

		mNewDuration = settings.scope.duration;
		mView.setProperty(\visible, ss.isVisible, true);
		mView.refresh;
	} /* stash */

	fetch { | settings |
		var ss;

		if (settings.waitingForStash, {
			this.stash(settings);
		});

		ss = settings.sampen;
		ss.amplitudeTolerance = mNumberBoxToleranceAmplitude.value;
		ss.amplitudeWindowSize = mNumberBoxWindowSizeAmplitude.value;
		ss.amplitudeSequenceLength = mNumberBoxSequenceLengthAmplitude.value;
		ss.amplitudeHarmonics = mNumberBoxHarmonicsAmplitude.value;

		ss.phaseTolerance = mNumberBoxTolerancePhase.value;
		ss.phaseWindowSize = mNumberBoxWindowSizePhase.value;
		ss.phaseSequenceLength = mNumberBoxSequenceLengthPhase.value;
		ss.phaseHarmonics = mNumberBoxHarmonicsPhase.value;

		ss.bDrawSampEn = mCheckBoxDrawSampEn.value;
		ss.bDrawQci = mCheckBoxDrawQci.value;
		ss.bDrawDEGGmax = mCheckBoxDrawDEGGmax.value;
		ss.bDrawCPP = mCheckBoxDrawCPP.value;
//		ss.bDrawIc = mCheckBoxDrawIc.value;
//		ss.bDrawCrest = mCheckBoxDrawCrest.value;
		ss.bDrawSpecBal = mCheckBoxDrawSpecBal.value;

		ss.isVisible = mView.visible;

		// Update the maximum SampEn measurement:
		value {
			var max1 =
			ss.amplitudeHarmonics *
			SampleEntropyFromBus.upperBounds(
				ss.amplitudeWindowSize,
				ss.amplitudeSequenceLength
			);

			var max2 =
			ss.phaseHarmonics *
			SampleEntropyFromBus.upperBounds(
				ss.phaseWindowSize,
				ss.phaseSequenceLength
			);

			var maxSampEn = max1 + max2;
			mScopeViewer.vspec = ControlSpec(nMinSampleEntropy, maxSampEn, units: " ")
		};

		mTempDuration = (mNewDuration*mXscale).clip(1,10);
		settings.scope.duration = mNewDuration;
	} /* fetch */

	updateData { | data |
		var scopeData = data.scope.sampen;
		var duration = mTempDuration;
		var dsg = data.settings.general;
		var dss = data.settings.sampen;


		if (dsg.guiChanged, {
			// Set the theme colors
			mView.background_(dsg.getThemeColor(\backPanel));
			mView.allChildren.do ({ arg c;
				if (c.isKindOf(StaticText), { c.stringColor_(dsg.getThemeColor(\panelText)) });
				// class CheckBox does not implement .stringColor (!!)
				if (c.isKindOf(CheckBox), {
					c.palette = c.palette.windowText_(dsg.getThemeColor(\panelText));
					c.palette = c.palette.window_(dsg.getThemeColor(\backPanel));
				});
			});

			mScopeViewer.background_(dsg.getThemeColor(\backGraph));
			mScopeViewer.gridFontColor_(dsg.getThemeColor(\panelText));
			this.updateGraphOnly;
		});

		this.enableInterface(data.general.started.not);

		if ( duration != mScopeViewer.hspec.range, {
			mScopeViewer.hspec = ControlSpec(duration.neg, 0, units: "s");
		});

		// Build the array of time-series for plotting, and draw them
		if ( scopeData.notNil and: data.general.started, {
			var series;

			series = dss.graphsRequested collect: { |type, i|
				case
				{ type == \Qcontact} { scopeData[i+1]*10 }
				{ type == \Icontact} { scopeData[i+1]*10 }
				{ type == \DelayedSpecBal} { (scopeData[i+1]+40)*0.25 }
				{ scopeData[i+1] }
			};
			mScopeViewer.update(scopeData.first, series, (data.general.pause == 2));
		});

		// Tell the ScopeViewer which colours we will be using
		if ( data.general.starting, {
			var c = [];
			[
				mCheckBoxDrawSpecBal,
				mCheckBoxDrawCPP,
				mCheckBoxDrawSampEn,
				mCheckBoxDrawQci,
				mCheckBoxDrawDEGGmax
//				mCheckBoxDrawIc,
//				mCheckBoxDrawCrest,
			] do: { | cb, i |
				if (cb.value, { c = c.add(mCurveColors[(i-3).mod(5)]) })
			};
			mScopeViewer.reset;
			mScopeViewer.colors = c;
		});

		if ( data.general.stopping, {
			mScopeViewer.stop;
		});
	} /* updateData */

	updateGraphOnly {
		mScopeViewer.refresh;
	}

	close {
	}
}