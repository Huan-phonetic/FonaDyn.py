// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //

VRPViewCluster {
	// Views
	var mView;
	var mViewClusterCentroids;
	var mViewClusterStats;

	// Controls
	var mButtonInit;
	var mButtonLearn;
	var mButtonReset;
	var mButtonLoad;
	var mButtonSave;
	var mStaticTextCluster;
	var mNumberBoxClusters;
	var mStaticTextHarmonics;
	var mNumberBoxHarmonics;
	var mStaticTextClusters;
	var mSliderCluster;

	// States
	var mSelected;
	var mStarted; // We need to know when the server is started, just started/stopped, and by keeping a member we can do that.
	var mResetNow; // We need to know when the server should reset the counts/centroids.
	var mCountDownFrames;  // The number of GUI updates to wait in this run before Auto Reset
	var mFramesWaited;
	var bRequestSaveSettings;
	var mbNormalize;
	var mbRealloc;

	// Data
	var mClusterCounts;
	var mClusterCentroids;  // the array
	var mCurrentCluster;
	var mSigmaApproxCoeffs; // Coeffs optionally used for unrippling the FD-derived wave shapes
	var mcDelim;

	var mDrawCycleData; // True if it should draw the cycle data, false otherwise.
	var mDrawableClusterCycle; // Object handling the drawing of the cluster cycle

	// Other GUI items
	var mPaletteFactory;
	var mPalette;
	var mbPaletteChangeNeeded;
	var mFont;
	var mGlyphs;
	var mStatsHeight;
	var mGridCentroids;
	var mSynthCurves; 	// array of polylines, one for each clustered waveform
	var mCycleCurveMax;

	var mLoadedData;
	var mClusterLoadedData;
	var mLoadedClusterSettings;
	var mClusterSettingsPathName;
	var mLastPath;		// path where _clusters.csv was last saved or loaded

	// Constants
	var <metricID;

	classvar <mAdapterUpdate;

	classvar iRelearn = 0;
	classvar iPrelearned = 1;

	classvar iLearn = 0;
	classvar iDontLearn = 1;

	classvar iReset = 0;
	classvar iAutoReset = 1;

	classvar iLoad = 0;
	classvar iUnload = 1;

	// GUI settings
	classvar nMinDb = -60;
	classvar nMaxDb = 0;
	classvar nResynthSteps = 80;  // Could take this from VRPMovingEGG

	classvar nDefaultStatsHeight = 125;

	*initClass {
		mAdapterUpdate = { "boo".postln };
	}

	*new { | view |
		^super.new.init(view);
	}


	update { | menu, who, what, newValue |
		switch (what,
			\normalization,
			{
				mbNormalize = newValue.asBoolean;
				this.remakeSynthCurves(mbNormalize);
			}
		);
	}

	remakeSynthCurves { arg bNorm;
		mSynthCurves = List.newClear(VRPSettingsCluster.nMaxClusters);
		mClusterCounts.size do: { | iC |
			var shape = this.makeSynthCurve(iC, nResynthSteps, bNorm);
			mSynthCurves.put(iC, shape);
		};
	}

	setPalette { | whichCluster, howManyClusters |
		mPaletteFactory.setClusters(whichCluster, howManyClusters);
		mPalette = mPaletteFactory.palette;
	}

	init { | view |
		var gridFont = Font.new(\Arial, 8, usePointSize: true);
		var gridYellow = Color.new(0.7, 0.7, 0);
		mView = view;
		metricID = VRPSettings.iClustersEGG;
		mLastPath = thisProcess.platform.recordingsDir;
		bRequestSaveSettings = false;
		mAdapterUpdate = { | menu, who, what, newValue |
			this.update(menu, who, what, newValue);
		};

		this addDependant: VRPViewMaps.mAdapterUpdate;

		mPaletteFactory = VRPSettings.metrics[VRPSettings.iClustersEGG].deepCopy;

		// View that shows the centroids (levels & phases) of the EGG clusters
		mViewClusterCentroids = UserView(mView, mView.bounds);
		mViewClusterCentroids
		.background_(Color.black)
		.drawFunc_{ | uv | this.drawCentroids(uv); };

		// Allow dropping of _cEGG.csv files onto the centroids plot
		mViewClusterCentroids.canReceiveDragHandler_({|v, x, y|
			var str, bOK = false;
			str = v.class.currentDrag;
			if ((mButtonLoad.value == iLoad) and: (str.class == String), {
				if (VRPDataCluster.testSuffix(str), {
					bOK = true;
				});
			} );
			bOK
		})
		.receiveDragHandler_({|v, x, y|
			var str, bOK = false;
			str = v.class.currentDrag;
			this.loadClusterSettingsPath(str);
			mButtonLoad.value = iUnload;
		});


		// View that shows the bar graph of the cluster point counts,
		// or the EGG wave shapes.
		mViewClusterStats = UserView(mView, mView.bounds);
		mViewClusterStats
		.background_(Color.black)
		.drawFunc_{ | uv |
			if ( mDrawCycleData and: mPalette.notNil,
				{
					if (mSelected > 0, {
						mDrawableClusterCycle.index_(mSelected-1);
						mDrawableClusterCycle.draw(uv, mPalette.(mSelected-1));
						this.drawResynthEGG(uv, mSelected-1, 0);
					},{
						var curve, yScale=1.0, b, h;
						this.drawResynthEGG(uv, mCurrentCluster, mClusterCounts.size);

						// Draw the running average curve for this cluster,
						// but scale it first
						curve = mSynthCurves.at(mCurrentCluster);
						if (curve.notNil, {
							yScale = curve[1]/mCycleCurveMax;
							b = uv.bounds.moveTo(0,0);
							h = b.height * yScale;
							b = Rect(b.left, b.bottom-h, b.width, h);
						});
						mDrawableClusterCycle.index_(mCurrentCluster);
						mDrawableClusterCycle.draw(uv, mPalette.(mCurrentCluster), b);
					})
				}, {
					this.drawStats(uv);
				}
			);
		};
		mViewClusterStats.canReceiveDragHandler = mViewClusterCentroids.canReceiveDragHandler;
		mViewClusterStats.receiveDragHandler = mViewClusterCentroids.receiveDragHandler;

		mClusterLoadedData = nil;
		mLoadedClusterSettings = nil;
		mDrawCycleData = false;
		mDrawableClusterCycle = DrawableClusterCycle(0, 0);
		mSigmaApproxCoeffs = nil ;
		mbNormalize = true;
		mCycleCurveMax = 1.0;

		mGridCentroids =
		DrawGrid(Rect(),
			ControlSpec(-1.0, 1, units: "π").grid,
			ControlSpec(nMinDb.asInteger, nMaxDb.asInteger, default: 0, units: "dB").grid
		)
		.font_(gridFont)
		.fontColor_(gridYellow)
		.smoothing_(false)
		.gridColors_([gridYellow, gridYellow]);
		if (Main.versionAtLeast(3,13),
			{
				mGridCentroids.tickSpacing_(50,25);
				mGridCentroids.x
				.labelAnchor_(\bottomLeft)
				.labelAlign_(\left)
				.labelOffset_(2@0)
				.labelsShowUnits_(true);
				mGridCentroids.y
				.labelAnchor_(\topLeft)
				.labelAlign_(\left)
				.labelOffset_(2@0)
				.labelsShowUnits_(true)
				.constrainLabelExtents_(false);
			}, {
				mGridCentroids.x.labelOffset.y = mGridCentroids.x.labelOffset.y * VRPMain.screenScale.y * 1.2;
			}
		);


		mGlyphs = this.makeGlyphs(10);

		mClusterCounts = nil;
		mClusterCentroids = nil;
		mSynthCurves = List.newClear(VRPSettingsCluster.nMaxClusters);
		mCurrentCluster = 0; // avoid nil, to prevent certain palette errors
		mbPaletteChangeNeeded = false;
		mbRealloc = false;
		mFont = Font("Arial", 8, true, false, true);
		mStatsHeight = nDefaultStatsHeight;
		mStarted = false;
		mResetNow = false;
		mCountDownFrames = -1;
		mcDelim = VRPMain.cListSeparator ;	// column separator in CSV files - locale-dependent

		this.initMenu();
		this.setPalette (0, mNumberBoxClusters.value);

		// Click on a colored column to select that cluster
		// Click in the upper third of the window to display all clusters
		// Hold down Ctrl and click left or right on a column to shift the cluster order
		mViewClusterStats.mouseDownAction_{
			| uv, x, y, mod, buttonNumber |
			if ( mClusterCounts.notNil, {
				var idx;
				idx = ((x / uv.bounds.width) * mClusterCounts.size).asInteger;
				if ((mDrawCycleData.not and: mod.isCtrl), {
					switch (buttonNumber,
						0, { this.shiftorder(idx, -1) },
						1, { this.shiftorder(idx,  1) },
						2, { this.sortorder }
					);
				},{
					if ((y < (uv.bounds.height/3)) or: mDrawCycleData,
						{
							idx = 0;
							mSliderCluster.valueAction_(idx);
						},
						{
							mDrawableClusterCycle.index = idx;
							mSliderCluster.valueAction_((idx+1)/mClusterCounts.size);
						}
					);
					mDrawCycleData = mDrawCycleData.not;
				})
			});
		};

		mView.layout_(
			VLayout(
				[
					HLayout(
						[mButtonInit, stretch: 1],
						[mButtonLearn, stretch: 1],
						[mButtonReset, stretch: 1],
						[mButtonLoad, stretch: 1],
						[mButtonSave, stretch: 1]
					), stretch: 2
				],

				[
					HLayout(
						[mStaticTextClusters, stretch: 1],
						[mNumberBoxClusters, stretch: 1],
						[mStaticTextHarmonics, stretch: 1],
						[mNumberBoxHarmonics, stretch: 1],
						[mStaticTextCluster, stretch: 1],
						[mSliderCluster, stretch: 5]
					), stretch: 2
				],

				[
					HLayout(
						[mViewClusterCentroids, stretch: 1],
						[mViewClusterStats, stretch: 1]
					), stretch: 2
				] // Force the menu to take up as little space as possible!
			)
		);
		mView.layout.margins_(5);
	}

	initMenu {
		var static_font = Font(\Arial, 8, usePointSize: true);

		////////////////////////////////////////////////////////////////////

		mButtonInit = Button(mView, Rect())
		.enabled_(false)
		.states_([
			["Init: Relearn"],
			["Init: Pre-learned"]
		])
		.action_({ | btn |
			this.updateMenu;
		})
		.canReceiveDragHandler_({ |v| v.class.prClearCurrentDrag; });

		/////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////

		mButtonLearn = Button(mView, Rect())
		.enabled_(false)
		.states_([
			["Learning: On"],
			["Learning: Off"]
		])
		.action_({ | btn |
			this.updateMenu;
		})
		.canReceiveDragHandler_({ |v| v.class.prClearCurrentDrag; });

		/////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////

		mButtonReset = Button(mView, Rect())
		.states_([ 	["Reset Counts"], ["Auto Reset", Color.black, Color.hsv(0.12, 1, 1)] ])
		.mouseDownAction_( {
			if ( mStarted, {
				// The server is started - so let it deal with the reset
				mResetNow = true;
			},{
				// The server is not started - so lets deal with it locally.
				if ( mClusterCounts.notNil, {
					mClusterCounts.fill(0);
				});
			})
		} )
		.canReceiveDragHandler_({ |v| v.class.prClearCurrentDrag; });


		/////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////

		mButtonLoad = Button(mView, Rect())
		.states_([
			["Load Clusters"],
			["Unload"]
		])
		.action_({ | btn |
			var load = btn.value == iUnload;
			if ( load,
				{
					this.loadClusterSettingsDialog;
				}, {
					mClusterCounts = nil;
					mClusterCentroids = nil;
					mSynthCurves = List.newClear(VRPSettingsCluster.nMaxClusters);
					mDrawableClusterCycle.clear; // Old averaged cycles no longer apply
					mCurrentCluster = 0; // avoid nil, to prevent certain palette errors
				}
			);
			this.updateMenu;
		})
		.canReceiveDragHandler_({|v, x, y|
			var str, bOK = false;
			str = v.class.currentDrag;
			if ((v.value == iLoad) and: (str.class == String), {
				if (VRPDataCluster.testSuffix(str), {
					bOK = true;
				} , {
					"Invalid cluster data.".warn;
				})
			} );
			bOK
		})
		.receiveDragHandler_({|v, x, y|
			var str, bOK = false;
			str = v.class.currentDrag;
			this.loadClusterSettingsPath(str);
			v.value = iUnload;
		});

		/////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////

		mButtonSave = Button(mView, Rect())
		.enabled_(false)
		.states_([
			["Save Clusters"]
		])
		.action_({
			bRequestSaveSettings = true;
			// Will invoke this.saveClusterSettingsDialog;
		})
		.canReceiveDragHandler_({ |v| v.class.prClearCurrentDrag; });


		/////////////////////////////////////////////////////////////////////

		mStaticTextClusters = StaticText(mView, Rect(0, 0, 100, 0))
		.string_("EGG clusters:")
		.font_(static_font);
		mStaticTextClusters
		.fixedWidth_(mStaticTextClusters.sizeHint.width)
		.fixedHeight_(20)
		.stringColor_(Color.white);

		mNumberBoxClusters = NumberBox(mView, Rect(0, 0, 100, 35))
		.value_(5)							// This is the default value
		.clipLo_(2)
		.clipHi_(VRPSettingsCluster.nMaxClusters)
		.step_(1)
		.scroll_step_(1)
		.fixedWidth_(24);

		/////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////

		mStaticTextHarmonics = StaticText(mView, Rect(0, 0, 100, 0))
		.string_("Harmonics:")
		.font_(static_font);
		mStaticTextHarmonics
		.fixedWidth_(mStaticTextHarmonics.sizeHint.width)
		.fixedHeight_(20);
		mStaticTextHarmonics.stringColor_(Color.white);

		mNumberBoxHarmonics = NumberBox(mView, Rect(0, 0, 100, 35))
		.value_(10)							// This is the default value
		.clipLo_(2)
		.clipHi_(VRPSettingsCluster.nMaxHarmonics)
		.step_(1)
		.scroll_step_(1)
		.fixedWidth_(24)
		.action_({
			this.updateMenu;
		});
		mNumberBoxHarmonics.align(\center);

		/////////////////////////////////////////////////////////////////////

		mStaticTextCluster = TextField(mView, [0, 0, 100, 20]);
		mStaticTextCluster
		.resize_(4)
		.align_(\right)
		.fixedWidth_(24)
		.enabled_(false);

		mSliderCluster = Slider(mView, [0, 0, mView.bounds.width, 30]);
		mSliderCluster
		.resize_(4)
		.maxHeight_(30)
		.value(0.0);

		// Ignore default keys if Alt is pressed
		mSliderCluster.keyDownAction = { |v, char, mod, unicode, keycode, key|
			var ret_val = nil;
			if (mod.isAlt, { ret_val = false },
				{ ret_val = v.defaultKeyDownAction(char, mod, unicode, keycode, key)}
			);
			ret_val
		};


		/////////////////////////////////////////////////////////////////////
		// Initialize the GUI members from global constants
		/////////////////////////////////////////////////////////////////////

		mSelected = 0;
		this.updateMenu;
	} /* initMenu */

	updateMenu {
		if ( mStarted.not, { // All disabled while it is started

			// Update the slider
			var req_step = 1 / (mNumberBoxClusters.value);

			if ( mSliderCluster.step != req_step, {
				mSliderCluster
				.step_(req_step)
				.action_{
					mSelected = (mSliderCluster.value*mNumberBoxClusters.value).round(1).asInteger;

					// Update the cluster text
					mStaticTextCluster.string_(
						// "Cluster: " ++
						if ( mSelected == 0, "All", mSelected.asString )
					);

					// Signal change of selected cluster
					this.changed(this, \selectCluster, mSelected);
				};

				// Reset the slider to 0 since it is no longer valid
				mSliderCluster.valueAction_(0);
				mSelected = 0;
			});

			// Enable/Disable depending on current states
			mButtonLoad.enabled_(true); // Load is always available

			switch (mButtonLoad.value,
				iLoad, {
					mButtonInit
					.enabled_(false)
					.value_(iRelearn); // Must relearn without any prelearned data

					// Can update the # of harmonics or clusters when nothing is loaded
					mNumberBoxHarmonics
					.enabled_(true);
					mNumberBoxClusters
					.enabled_(true);
				},

				iUnload, {
					mButtonInit
					.enabled_(true); // May choose to use prelearned data or not

					// Cannot update the # of harmonics or clusters since we have data loaded
					mNumberBoxHarmonics
					.enabled_(false);
					mNumberBoxClusters
					.enabled_(false);
				}
			);

			switch (mButtonInit.value,
				iRelearn, {
					mButtonLearn
					.enabled_(false)
					.value_(iLearn); // Must learn with relearn active
				},

				iPrelearned, {
					mButtonLearn
					.enabled_(true); // May choose to continue learning or not

					// Cannot update the # of harmonics or clusters with prelearned data
					mNumberBoxHarmonics
					.enabled_(false);
					mNumberBoxClusters
					.enabled_(false);
				}
			);

			// Can only use reset while learning is on
			mButtonReset
			.enabled_( mButtonLearn.value == iLearn );

			// Cannot save without any data
			mButtonSave
			.enabled_( mButtonLoad.value == iUnload );
		});
	} /* updateMenu */

	saveClusterSettingsDialog { arg cSettings;
		VRPMain.savePanelPauseGUI({ | path |
			mLastPath = PathName.new(path).pathOnly;
			cSettings.saveClusterSettings(path);
			mClusterSettingsPathName = path;
		}, path: mLastPath, wantedSuffix: VRPDataCluster.csvSuffix[\EGGcsvSuffix]);
	}

	makeGlyphs{ arg nHarmonics;
		var g;
		g = (nHarmonics - 1) collect: { | n | (n + 2).asString };
		g = g.add ("♦");  // Indicates the fundamental's absolute level (dBFS) and phase
		^g
	}

	loadClusterSettingsPath{ arg inPath=nil;
		var c, h, chosenPath;
		var tempClusterSettings = VRPSettingsCluster.new(nil);
		if (inPath.notNil, {
			chosenPath = inPath;

			#c, h = tempClusterSettings.loadClusterSettings(chosenPath);
			mClusterSettingsPathName = chosenPath;
			if (c >= 2, {
				mLoadedClusterSettings = tempClusterSettings;   // Queue new settings for for .stash
				mLoadedClusterSettings.learn = false;
				mLastPath = PathName.new(chosenPath).pathOnly;
			}, {
				format("Could not parse the file %", PathName.new(chosenPath).fileName).error
			});
		});
	} /* loadClusterSettingsPath{} */

	loadClusterSettingsDialog {
		VRPMain.openPanelPauseGUI(
			{ | path |
				this.loadClusterSettingsPath(path);
			} , {
				mButtonLoad.value = iLoad; // Cancelled
		} , path: mLastPath);
	}

	reorder { arg newOrder;
		var tmp, bOK;
		bOK = true;
		if ((mStarted.not) and: (newOrder.class == Array) and: (newOrder.size == mClusterCounts.size),
			{
				newOrder.do { |elem, i| if (elem.class != Integer,  { bOK = false } )};
				if (bOK, {
					tmp = mClusterCounts[newOrder];
					mClusterCounts = tmp;
					tmp = mClusterCentroids[newOrder].deepCopy;
					mClusterCentroids = tmp;
					mDrawCycleData = false;
					mDrawableClusterCycle.clear; 	// New clusters invalidate the old averaged cycles
					mClusterSettingsPathName = nil; // Centroids no longer match those loaded from file
					this.changed(this, \reorderClusters, newOrder);
					this.remakeSynthCurves(mbNormalize);
					}
				)
		})
	}

	// Shift cluster iCluster by nSteps (< 0 left, > 0 right)
	shiftorder { arg iCluster, nSteps;
		var nC, kC;
		var newOrder;
		nC = mClusterCounts.size;
		kC = (iCluster + nSteps).mod(nC);
		newOrder = (0..nC-1).swap(iCluster, kC);
		this.reorder(newOrder);
	}

	sortorder {
		// SKETCHES
/*		var nC, nHarm;
		var cosX, sinX, phi;
		var phiSums, phiSorted;
		var oldOrder, newOrder;
		nC = mClusterCounts.size;
		nHarm = mNumberBoxHarmonics.value.asInteger;

		// Make an array containing for each cluster
		// the cluster number and the weighted sum of its phases
		phiSums = nC.collect( { |centroid|
			var phis;
			phis = (nHarm-1) collect: { | j |
				var wPhi;
				cosX = mClusterCentroids[ centroid ][ j + nHarm ];
				sinX = mClusterCentroids[ centroid ][ j + (nHarm*2) ];
				wPhi = atan2(sinX, cosX) * ((nHarm-j)/nHarm);  // Weigh higher harmonics as less
				wPhi
			};
			[ centroid, phis.sum ]
		});
phiSums.postln;

		// Now sort phiSums by descending angle-sum
		phiSorted = phiSums.sort({ arg a, b; a[1] > b[1] });
phiSorted.postln;
		newOrder = phiSorted.collect({ |e| e[0]});
newOrder.postln;
		this.reorder(newOrder); */
	}

	stash { | settings |
		var cs = settings.cluster;

		mClusterCounts = cs.pointsInCluster;
		mClusterCentroids = cs.centroids;
		mClusterSettingsPathName = cs.filePath;
		if (cs.filePath.notNil, {
			mButtonLoad.value_(iUnload);
		});
		if (mNumberBoxClusters.value != cs.nClusters, {
			mNumberBoxClusters.value = cs.nClusters;
			mbRealloc = true;
		});
		mNumberBoxHarmonics.value_(cs.nHarmonics);
		mGlyphs = this.makeGlyphs(cs.nHarmonics);
		mButtonInit.value_(cs.initialize.if (iPrelearned, iRelearn));
		mButtonLearn.value_(cs.learn.if (iLearn, iDontLearn));
		if (mClusterCentroids.notNil, {
			cs.nClusters do: { | iC |
				var shape = this.makeSynthCurve(iC, nResynthSteps, mbNormalize);
				mSynthCurves.put(iC, shape);
		}});
		mButtonReset.value_(cs.autoReset.if (iAutoReset, iReset));
		mView.setProperty(\visible, cs.isVisible, true);
	} /* .stash */

	fetch { | settings |
		var nCpre, nCpost;
		var cs = settings.cluster;
 		nCpre = cs.nClusters;

		case
		{ cs.pleaseStashThis.notNil}
		{	// A new _cEGG.csv file has been loaded by a script
			settings.cluster = cs.pleaseStashThis;
			this.stash(settings);
			mbRealloc = false;
			cs.pleaseStashThis = nil;
		}
		{ mLoadedClusterSettings.notNil }
		{	// A new _cEGG.csv file has been loaded by the user
			settings.cluster = mLoadedClusterSettings;
			this.stash(settings);
			mButtonLoad.value_(iUnload);
			mbRealloc = false;
			mLoadedClusterSettings = nil;
		}
		{ settings.waitingForStash() }
		{	// A new setting has been set from a script,
			// or all have been retrieved from the archive.
			this.stash(settings);
		}; /* case */

		{ cs.isVisible = mView.visible }.defer;
		cs.filePath = mClusterSettingsPathName;
		cs.pointsInCluster = mClusterCounts;
		cs.centroids = mClusterCentroids;
		cs.initialize = (mButtonInit.value == iPrelearned);
		cs.learn = (mButtonLearn.value == iLearn);
		cs.autoReset = (mButtonReset.value == iAutoReset);
		cs.nHarmonics_(mNumberBoxHarmonics.value.asInteger);
		settings.csdft.nHarmonics_(mNumberBoxHarmonics.value.asInteger);

		nCpost = mNumberBoxClusters.value.asInteger;
		if (nCpre != nCpost, {
			mbRealloc = true;
		});

		if (mbRealloc, {
			cs.allocCentroids(nCpost, cs.nHarmonics);
			mbPaletteChangeNeeded = true;
			this.changed(this, \numberOfClusters, nCpost);
			mbRealloc = false;
		});

		if (mbPaletteChangeNeeded, {
			this.setPalette(0, nCpost);
			mbPaletteChangeNeeded = false;
		});

		if (bRequestSaveSettings, {
			this.saveClusterSettingsDialog(cs);
			bRequestSaveSettings = false;
		});
	} /* fetch */

	updateData { | data |
		var cd = data.cluster;
		var cs = data.settings.cluster;
		var csg = data.settings.general;

		if (csg.guiChanged, {
			mView.background_(csg.getThemeColor(\backPanel));
			mViewClusterCentroids.background_(csg.getThemeColor(\backGraph));
			mViewClusterStats.background_(csg.getThemeColor(\backGraph));
			[mStaticTextClusters, mStaticTextHarmonics].do ({ arg c;
				c.stringColor_(csg.getThemeColor(\panelText))
			})
		});

		// NOTE: We check if the server is started with mStarted, since we rather care
		// about not missing out on data at the end, than starting to grab data at the
		// first possible chance.
		if (mStarted, {
			// Grab the newly updated data
			mClusterCounts = cd.pointsInCluster;
			mClusterCentroids = cd.centroids;
			mCurrentCluster = data.vrp.currentCluster ? 0;
			mDrawableClusterCycle.data = cd.cycleData;

			// Wait until the clarity has been above threshold
			// for "iFramesToReset" consecutive GUI updates,
			// and then issue an Auto Reset of the clusters
			if ((mCountDownFrames > 0) and: data.vrp.currentClarity.notNil,
				{	// Increment if clarity is above threshold, otherwise start over
					if (data.vrp.currentClarity >= VRPSettings.metrics[VRPSettings.iClarity].minVal,
						{ mFramesWaited = mFramesWaited + 1 } , { mFramesWaited = 0 }
					);
					if (mFramesWaited >= mCountDownFrames,
						{
							mResetNow = true;
							mCountDownFrames = 0;
							mButtonReset.value = iReset;
							Date.localtime.format("RESET EGG clusters %H:%M:%S").postln;
						}
					);
				}
			);

			if ( mResetNow, {
				cd.resetNow = true;
			});
		});

		mResetNow = false;

		if (mStarted.not and: data.general.started, {
			var nGlyphs = cs.nHarmonics.asInteger;
			mGlyphs = this.makeGlyphs(nGlyphs);

			mSigmaApproxCoeffs = cs.suppressGibbs.if
			{ Array.fill(nGlyphs, { arg i; 	var c = ((i+1.0)/(nGlyphs+1))*pi; sin(c)/c })}
			{ nil };

			// Update the # of clusters/samples in DrawableClusterCycle
			// Just started the server - so forcefully disable all input controls except reset
			[
				mNumberBoxClusters,
				mNumberBoxHarmonics,
				mButtonInit,
				mButtonLearn,
				mButtonLoad,
				mButtonSave
			]
			do: { | x | x.enabled_(false); };

			mDrawableClusterCycle.count = cs.nClusters;
			mDrawableClusterCycle.samples = cs.nSamples;

			// Arm the Auto Reset mechanism
			if (cs.autoReset,
				{
					mCountDownFrames = cs.iFramesToReset;
					mFramesWaited = 0;
				}
			);

			// Clear the file name if centroid data will change
			if (mButtonLearn.value == iLearn, {
				mClusterSettingsPathName = nil;
			});

			// Reset the mStatsHeight
			mStatsHeight = nDefaultStatsHeight;

			// Clear the list of drawn cycles
			mSynthCurves = List.newClear(VRPSettingsCluster.nMaxClusters);
		});

		if (mStarted and: data.general.started.not, {
			// Just stopped the server
			cd.resetNow = false; // It shouldn't matter leaving it as true, but we do this for safety.
			mButtonLoad.value_(iUnload); // Have data since we just stopped the server
		});

		mStarted = data.general.started;
		this.updateMenu;
		mViewClusterCentroids.refresh;
		mViewClusterStats.refresh;
	} /* updateData */

	drawCentroids { | view |
		var seq, bounds;

		// Draw the grid first
		mGridCentroids.bounds_( view.bounds.moveTo(0, 0).insetAll(0, 0, 1, 1) );
		mGridCentroids.draw;

		// Draw the glyphs after
		if (mClusterCentroids.notNil, {
			bounds = mGridCentroids.bounds;

			if (0 == mSelected, {
				seq = ((mNumberBoxClusters.value-1)..0);
			}, {
				seq = [mSelected - 1];
			});
			Pen.use {
				var nGlyphs = mNumberBoxHarmonics.value.asInteger;
				seq do: { | i |
					var cosX, sinX, x, y, xPix, yPix, color;
					color = mPalette.(i);
					nGlyphs do: { | j |
						cosX = mClusterCentroids[ i ][ j + nGlyphs ];
						sinX = mClusterCentroids[ i ][ j + (nGlyphs*2) ];
						// Un-weighting of cos and sin is not needed
						x = atan2(sinX, cosX);
						xPix = x.linlin(-pi, pi, 0, bounds.width);
						y = mClusterCentroids[ i ][ j ];
						// Unweighting of L0 is needed
						if (j==(nGlyphs-1), { y = y*1000.0 });
						yPix = y.linlin(nMinDb*0.1, 0, bounds.height, 0);  // -6 Bel or -60 dB
						mGlyphs[j].drawAtPoint(xPix@yPix, mFont, color);
						if (mSelected > 0, {
							Pen.fillColor = color;
							Pen.fillOval(Rect(xPix,yPix,0,0).insetBy(-1));
						});
					};
				};
				"1".drawAtPoint(bounds.width.half @ 0, mFont, Color.gray(0.7));
			};
		});
	}

	drawStats { | view |
		var rc, bounds, barWidth, fColor, max = 50;
		var str, counts;

		if (mClusterCounts.notNil, {
			str = format(" % cycles", mStatsHeight.asString);
			counts = mClusterCounts;
			max = mClusterCounts.maxItem;
		});

		if (mNumberBoxClusters.hasFocus, {
			str = format(" Palette for % EGG waveshape clusters", mNumberBoxClusters.value.asInteger);
			counts = max ! mNumberBoxClusters.value;
			mStatsHeight = nDefaultStatsHeight;
			max = 50;
		});

		if (mClusterSettingsPathName.notNil
			and: (mButtonLearn.value == iDontLearn)
			and: (mButtonInit.value == iPrelearned),
			{
				str =  format(" %", mClusterSettingsPathName.basename);
			}
		);

		// Update the virtual height of the stats view
		while ( { mStatsHeight < max }, { mStatsHeight = mStatsHeight * 2; } );

		if (counts.notNil, {
			bounds = view.bounds;
			rc = Rect();
			barWidth = bounds.width / mNumberBoxClusters.value;
			Pen.use{
				str.drawAtPoint(0@0, mFont, Color.yellow(0.7));
				mNumberBoxClusters.value.do ({ | i |
					var xPix, yPix, count;
					count = counts[i]; // mClusterCounts[i];
					xPix = i.linlin(0, mNumberBoxClusters.value, 0, bounds.width);
					yPix = count.linlin(0, mStatsHeight, 0, bounds.height);
					rc.set(xPix, bounds.height - yPix, barWidth, yPix);
					Pen.fillColor = mPalette.(i);
					Pen.fillRect(rc);
				});
			};
		});
	}

	drawResynthEGG { arg uView, clustNr, overlays;
		var shape, list, vals, s, sMax, shapeWithMax;
		var errHeight = DrawableClusterCycle.errorSpace;
		var b = uView.bounds.moveTo(0,0);
		var nDeltas = mNumberBoxHarmonics.value.asInteger;
		var nSteps = nResynthSteps;
		var dEGGscale = (0.5*sin(2pi/nSteps)).reciprocal;

		// First resynthesize the EGG curve for the current cluster only
		// Then save this curve into the list mSynthCurves
		if (mClusterCounts.notNil && mClusterCentroids.notNil, {
			shapeWithMax = this.makeSynthCurve(clustNr, nSteps, mbNormalize);
			mSynthCurves.put(clustNr, shapeWithMax);
		});


		// Now draw the cycle curves in the list mSynthCurves, one per cluster
		if (overlays > 0, { list = (0..overlays-1) }, { list = [clustNr] } );

		// Find the maximum amplitude in all curves
		mCycleCurveMax = 1.0;
		if (mbNormalize.not, {
			vals = list.collect( { | curveNo, ix |
				var c, val;
				c = mSynthCurves.at(curveNo.asInteger);
				val = if (c.notNil) { c[1] } { 1.0 } ;
				// format("%: val=%;  ", curveNo, val.round(0.001)).post;
				val
			});
			mCycleCurveMax = vals.maxItem;
			// ("max ="+mCycleCurveMax).postln;
		});

		// Draw all the curves in list
		list.do { arg index;
			if ((s = mSynthCurves.at(index)).notNil, {
				shape = s[0].normalize(0, s[1]/mCycleCurveMax);
				Pen.use {
					var pStr, normStr = "";
					var color = mPalette.(index);
					if (clustNr == index, {
						if (mbNormalize, {
							normStr = "Normalized";
						});
						if ((list.size == 1) or: mbNormalize, {
							var qCI = shape[1..nSteps].sum * nSteps.reciprocal;
							var maxIx = shape[1..nSteps].maxIndex({ |item, ix| item - shape[ix] });
							var dEGGmaxN = (shape[maxIx+1] - shape[maxIx]) * dEGGscale;

							("Qci " ++ qCI.asStringPrec(3).padRight(5, "0"))
							.drawLeftJustIn(Rect(3, b.height - errHeight, b.width, errHeight), mFont, color);

							("QΔ " ++ dEGGmaxN.round(1e-2).asString.padRight(4, "0"))
							.drawRightJustIn(Rect(0, b.height/2.2, b.width-3, errHeight*2), mFont, color);
						});
						pStr = if (mSigmaApproxCoeffs.isNil,
							{ "% Sum of % harmonics" },
							{ "% σ-approx of % harmonics" }
						);
						format(pStr, normStr, nDeltas).drawRightJustIn(Rect(0, 0, b.width-2, errHeight), mFont, color);
						Pen.width = 0.02
					},{
						Pen.width = 0.01
					});
					Pen.strokeColor = color ? Color.grey;
					Pen.translate(b.left, b.top);
					Pen.translate( 0.0, errHeight );
					Pen.scale( b.width / nSteps, errHeight - b.height);
					Pen.translate( 0, -1);

					// Draw the lines
					Pen.moveTo(Point(0,shape[0]));
					shape do: { |v, i| Pen.lineTo(i@v) };
					Pen.stroke;
				};
			});
		}
	} /* drawResynthEGG */


	makeSynthCurve { | clustNr, nSteps, bNormalize=true |
		var pulse, shape, amps, phases, phase1;
		var nDeltas = mNumberBoxHarmonics.value.asInteger;
		var mCC = mClusterCentroids[clustNr];
		var bels, bels0, myMax;

		bels0 = if (bNormalize, 0.0, { mCC[nDeltas-1]*1000.0 } );
		bels = [0.0] ++ (nDeltas-1).collect( { |i| mCC[i] } );
		amps = dbamp((bels+bels0) * 10.0);

		// Apply sigma approximation to minimize Gibbs' phenomenon.
		// This looks much better, but reduces the displayed Qdeltas even further.
		if (mSigmaApproxCoeffs.notNil, { amps = amps * mSigmaApproxCoeffs } );

		phases = nDeltas collect: { |i| atan2(mCC[i + (nDeltas*2)], mCC[i + nDeltas]) };
		phase1 = phases[nDeltas-1] + 0.5pi; // add pi/2 to use .addSine instead of cos
		phases = [0.0] ++ phases + phase1;  // Reconstruct actual phases (not deltas)

		// Don't use Signal.fillSine() - it normalizes differently
		pulse = Signal.newClear(nSteps);
		// Ignore the last element in amps[] and phases[]
		nDeltas do: { arg i; pulse.addSine(i+1, amps[i], phases[i]) } ;
		shape = FloatArray.newFrom(pulse); // .normalize;
		myMax = if (bNormalize, 1.0, { shape.maxItem } );
		shape = shape.add(shape[0]);
		^[shape, myMax]
	}

	close {
		this.release;
	}
}

