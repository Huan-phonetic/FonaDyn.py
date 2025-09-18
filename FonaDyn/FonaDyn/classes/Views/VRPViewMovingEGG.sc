// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //

VRPViewMovingEGG {
	var mView;

	var mButtonNormalize;
	var mbNormalize;
	var mStaticTextCount;
	var mNumberBoxCount;
	var mStaticTextSamples;
	var mNumberBoxSamples;
	var mStaticTextMethod;
	var mUV;
	var mDMEGG;
	var mStaticTextThreshold, mNumberBoxThreshold;
	var mScopePenColor;
	var strokeCount;

	*new { | view |
		^super.new.init(view);
	}

	init { | view |
		var static_font = Font.new(\Arial, 8, usePointSize: true);
		var bold_font = Font.new(\Arial, 8, true, false, true);

		mView = view;
		mbNormalize = true;
		mScopePenColor = Color.gray(0.2);

		mButtonNormalize = Button(mView, Rect())
		.states_([
			["Normalize: Off"],
			["Normalize: On "]
		])
		.action_({ | v | this.changed(this, \normalization, v.value.asBoolean) })
		.value_(1);

		this addDependant: VRPViewCluster.mAdapterUpdate;

		///////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////

		mStaticTextThreshold = StaticText(mView, Rect())
		.string_("De-noise:")
		.font_(static_font);
		mStaticTextThreshold
		.fixedWidth_(mStaticTextThreshold.sizeHint.width)
		.fixedHeight_(35)
		.stringColor_(Color.white);

		mNumberBoxThreshold = NumberBox(mView, Rect())
		.value_(0.0)		// set default value here
		.clipLo_(0.0)
		.clipHi_(10.0)
		.step_(0.1)
		.align_(\center)
		.scroll_step_(0.05)
		.fixedWidth_(30);

		///////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////

		mStaticTextCount = StaticText(mView, Rect())
		.string_("Count:")
		.font_(static_font);
		mStaticTextCount
		.fixedWidth_(mStaticTextCount.sizeHint.width)
		.fixedHeight_(35)
		.stringColor_(Color.white);

		mNumberBoxCount = NumberBox(mView, Rect())
		.value_(5)
		.clipLo_(1)
		.clipHi_(50)
		.step_(1)
		.scroll_step_(1)
		.fixedWidth_(30);

		strokeCount = mNumberBoxCount.value;

		///////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////

		mStaticTextSamples = StaticText(mView, Rect())
		.string_("Samples:")
		.font_(static_font);
		mStaticTextSamples
		.fixedWidth_(mStaticTextSamples.sizeHint.width)
		.fixedHeight_(35)
		.stringColor_(Color.white);

		mNumberBoxSamples = NumberBox(mView, Rect())
		.value_(80)
		.clipLo_(1)
		.clipHi_(200)
		.step_(1)
		.scroll_step_(1)
		.fixedWidth_(30);

		mStaticTextMethod = StaticText(mView, Rect())
		.string_("ΦΛ")
		.align_(\left)
		.font_(bold_font);
		mStaticTextMethod
		.fixedWidth_(mStaticTextMethod.sizeHint.width)
		.fixedHeight_(35)
		.stringColor_(Color.white);

		mUV = UserView(mView, Rect())
		.background_(Color.white)
		.drawFunc_{
			if (mDMEGG.notNil, {
				mDMEGG.setCount(strokeCount);
				mDMEGG.draw(mUV);
			});
		};

		mView.layout_(
			VLayout(
				[
					HLayout(
						[mButtonNormalize, stretch: 1],
						[mStaticTextThreshold, stretch: 1],
						[mNumberBoxThreshold, stretch: 1],
						[mStaticTextCount, stretch: 1],
						[mNumberBoxCount, stretch: 1],
						[mStaticTextSamples, stretch: 1],
						[mNumberBoxSamples, stretch: 1],
						[mStaticTextMethod, stretch: 1, align: \right],
						[nil, stretch: 8]
					),
				stretch: 1],
				[mUV, stretch: 8]
			)
		);
		mView.layout.margins_(5);
	}

	stash { | settings |
		var ss = settings.scope;
		mButtonNormalize.value_(ss.normalize);
		mNumberBoxCount.value_(ss.movingEGGCount);
		mNumberBoxSamples.value_(ss.movingEGGSamples);
		mNumberBoxThreshold.value_(ss.noiseThreshold);
		mStaticTextMethod.string_(if (settings.csdft.method == VRPSettingsCSDFT.methodPhasePortrait, "Φ", "Λ"));
		mView.setProperty(\visible, ss.isVisible, true);
		// mView.visible_(ss.isVisible);
		this.changed(this, \normalization, ss.normalize);
	}

	fetch { | settings |
		var ss = settings.scope;

		if (settings.waitingForStash, {
			this.stash(settings);
		});

		ss.normalize = mButtonNormalize.value;
		mbNormalize = mButtonNormalize.value.asBoolean;
		ss.movingEGGCount = mNumberBoxCount.value;
		ss.movingEGGSamples = mNumberBoxSamples.value;
		ss.isVisible = (mView.visible);
		settings.scope.noiseThreshold = mNumberBoxThreshold.value;
	}

	updateData { | data |
		var sd = data.scope;
		var gd = data.general;
		var s = data.settings;
		var dsg = data.settings.general;
		var ss = s.scope;

		if (gd.starting, {
			if (mDMEGG.isNil, {
				mDMEGG = DrawableMovingEGG(ss.movingEGGCount, ss.movingEGGSamples, ss.normalize);
			});
			mDMEGG.penColor_(mScopePenColor);
			sd.busThreshold.set(mNumberBoxThreshold.value);  	// initialize it on the server
		});

		if (gd.stopping, {
			mDMEGG = nil;
			mNumberBoxThreshold.enabled_(true);
		});

		if (gd.started and: mDMEGG.notNil, {
			mDMEGG.data = sd.movingEGGData;
			mDMEGG.normalized_(mbNormalize);
			if (ss.noiseThreshold != mNumberBoxThreshold.value, {	// If the threshold was changed,
				sd.busThreshold.set(mNumberBoxThreshold.value);  	// send it to the server
				ss.noiseThreshold = mNumberBoxThreshold.value;
			});
			if ((data.vrp.currentClarity ? 0) < s.vrp.clarityThreshold,
				{
					strokeCount = max(0, strokeCount - 1);
				}, {
					strokeCount = mNumberBoxCount.value;
				}
			);
		});

		mStaticTextMethod.string_(if (s.csdft.method == VRPSettingsCSDFT.methodPhasePortrait, "Φ", "Λ"));

		// Noise thresholding must not be changed during fixed analysis
		mNumberBoxThreshold.enabled_(gd.started.not or: s.vrp.wantsContextSave.not);
		mNumberBoxThreshold.background_( mNumberBoxThreshold.enabled.if (Color.white, Color.gray));

		[
			mButtonNormalize,
			mNumberBoxSamples,
			mNumberBoxCount
		]
		do: { | x | x.enabled_(gd.started.not); };

		this.showDiagnostics(dsg.enabledDiagnostics);

		if (dsg.guiChanged, {
			mView.background_(dsg.getThemeColor(\backPanel));
			mUV.background_(dsg.getThemeColor(\backMap));
			mScopePenColor = (dsg.getThemeColor(\curveScope));
			mView.allChildren do: ({ arg c;
				if (c.isKindOf(StaticText), { c.stringColor_(dsg.getThemeColor(\panelText)) })}
			);
		});

		mUV.refresh;
	}

	showDiagnostics { | bShow |
		[
			mStaticTextCount,
			mNumberBoxCount,
			mStaticTextSamples,
			mNumberBoxSamples
		] do: { | b, i | b.visible_(bShow) };
	}

	close { nil }
}