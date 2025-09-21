// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //

VRPViewVRP{
	// VARIABLES //////////////////////////////
	// Views
	var mView;
	var mUserViewHolder;
	var mUserViewMatrix;
	var mUserViewMatrixBack;
	var mUserViewCursor;
	var mUserViewTarget;

	// Graphics
	var mDrawableSparseMatrix;
	var mDrawableSparseMatrixBack;
	var mClarityThreshold;
	var mGrid;
	var mGridFont;
	var mGridSpecMIDI, mGridSpecSPL;
	var mGridSpecFreq, mGridSpecHz;
	var mGridLinesMIDI, mGridLinesHz, mGridLinesSPL;
	var <mGridHorzGridSelectHz;
	var mColorRect;
	var mColorGrid, mColorGridFontColor;
	var mColorSpec;
	var mColorbarPalette;
	var mColorbarText;
	var fnFixedAspectLayout;
	var mHmargin, mVmargin;

	// Controls
	var mStaticTextLayer;
	var mDropDownType;
	var mSliderCluster;
	var mStaticTextCluster;
	var mStaticTextCycleThreshold;
	var mNumberBoxCycleThreshold; // min cycle count for display
	var mButtonLoadVRPdata;
	var mButtonSaveVRPdata;
	var mButtonSaveVRPimage;
	var mMapFileName;
	var mStaticTextInfo;

	// Entire VRP data
	var <>mVRPdata;
	var mVRPdataLoaded;

	// States
	var mMapMode;
	var mLayerSelected;
	var theLayer, theMetric;
	var mKeepNewest; 			// FALSE unless I am NOW and (TWIN exists and is showing the same layer)
	var mKeepNewestBack;		// FALSE unless I am NOW and (TWIN exists and is showing a cluster layer)
	var myMetrics;				// A local deepcopy of VRPSettings.metrics
	var mnClusters;				// The number of clusters in the current view, if any
	var mDictNclusters; 		// Two entries: the number of EGG clusters and the number of Phon clusters
	var stoppingTicks;

	var mClusterSelected;		// The cluster currently selected in this view
	var mDictSelections;		// Two entries: selected EGG cluster and selected Phon cluster
	var mbClosing;

	var mClusterPalette;
	var mCursorPalette;
	var mCursorColor;
	var mCursorRect;
	var mCellCount;
	var mbSized;
	var mbRedrawBack, mbRedrawFront, mbRedrawTarget, mbRedrawCursor;
	var mbMouseCellCoords;
	var mbMouseDownShift;
	var nMouseDownRepeats;
	var bRequestSetCentroid = 0;
	var <mLastPath;
	var mLargerFont;
	var mNewClusterOrder;
	var mSavedMapPathStr;
	var mbSaveContext;
	var mSignalDSM;

	classvar mCursorRectScaler;  // classvar, so as to be visible to multiple views
	classvar ptMouseDownLast;

	// CONSTANTS //////////////////////////////////////////////////

	// Map listening states
	classvar <iDisabled = 0;
	classvar <iUnavailable = 1;
	classvar <iReady = 2;
	classvar <iPlaying = 3;

	// Map types
	classvar <iNormal = 0;		// NOW
	classvar <iClone = 1;		// TWIN
	classvar <iCloneFloat = 2;	// TWIN FLOATING
	classvar <iReference = 3;	// BEFORE
	classvar <iDiff = 4;		// DIFF
	classvar <iSmooth = 5;      // SMOOTH

	classvar iCursorWidth = 9;
	classvar iCursorHeight = 9;

	// METHODS //////////////////////////////////////////////////

	*new { | view, vrpCopy=nil |
		^super.new.init(view, vrpCopy);
	}

	invalidate { | all |
		if (all and: mUserViewMatrixBack.notNil,
			{ mUserViewMatrixBack.clearOnRefresh_(true) });
		if (mUserViewMatrix.notNil,
			{ mUserViewMatrix.clearOnRefresh_(true) });
		if (mUserViewTarget.notNil,
			{ mUserViewTarget.clearOnRefresh_(true) });
	}

	setColorScale { arg layer, selCluster;
		var gridX;
		var m = myMetrics[layer];
		var sym = m.class.symbol;

		if (mVRPdata.notNil, // might be nil during init
			{
				m = mVRPdata.layers[sym].metric(selCluster);
			}
		);

		mColorSpec = ControlSpec(m.minVal, m.maxVal, m.colorBarWarpType, units: m.unit);
		// gridX = GridLines.new(mColorSpec);
		gridX = mColorSpec.grid;
		mColorGrid = DrawGrid(mColorRect, gridX, nil);
		mColorGrid
		.smoothing_(false)
		.font_(mGridFont);

		if (Main.versionAtMost(3,12,2),

			// ====== To SC v3.12.2 ===============
			{
				mColorGrid.x.labelOffset.x_(4)
			}, {

				// ====== From SC v3.13.0 rc 2 ========
				gridX.appendLabel_(mColorSpec.units);
				mColorGrid.x
				.labelAnchor_(\bottom)
				.labelAlign_(\left)
				.labelAppendString_(m.unit)		// seems to have no effect...
				.labelsShowUnits_(true)			// seems to have no effect...
				.labelOffset_(3@(-1));
			}
		);

	} /* setColorScale */

	setSPLscale {
		mGridSpecSPL = ControlSpec(VRPDataVRP.nMinSPL, VRPDataVRP.nMaxSPL, units: "dB");
		mGrid = DrawGrid(
			Rect(),
			if (mGridHorzGridSelectHz, mGridSpecHz.grid, mGridSpecMIDI.grid),
			mGridSpecSPL.grid
		);
		this.mapMode_(mMapMode); // To set grid colors after loadVRPdata*

		if (Main.versionAtMost(3,12,2),
			{
				// ====== To SC v3.12.2 ========
				mGrid.x.labelOffset.y = mGrid.x.labelOffset.y * VRPMain.screenScale.y * 1.2;
				mGrid.y.labelOffset.y = -1;
			}, {
				// ====== From SC v3.13.0 rc 2 ========
				mGrid.x
				.labelAnchor_(\bottomLeft)
				.labelAlign_(\left)
				.labelOffset_(3 @ (mGrid.x.labelOffset.y * VRPMain.screenScale.y - 1))
				.labelAppendString_([" MIDI", " Hz"][mGridHorzGridSelectHz.asInteger]);
				mGrid.y
				.labelAnchor_(\topLeft)
				.labelAlign_(\left)
				.labelOffset_(3@0)
				.labelAppendString_(" dB")
				.constrainLabelExtents_(false);
			}
		);

		this.invalidate(true);
	} /* setSPLscale */

	mapSwitches {
		if (mClusterSelected.isNil, { "mClusterSelected was nil".warn} );  //// for debug only
		^[mLayerSelected, mnClusters ? 5, mClusterSelected ? 0, mMapFileName, mGridHorzGridSelectHz ]
	}

	mapMode { ^mMapMode }

	layers { ^[mLayerSelected, mClusterSelected] }

	setClarityThreshold { | t |
		if (mClarityThreshold != t, {
			mClarityThreshold = t;
			myMetrics[VRPSettings.iClarity].minVal_(t);
			if (mMapMode >= iReference, {
				var m = mVRPdata.layers[\Clarity].metric;
				m.minVal_(mClarityThreshold);
				this.setColorScale(VRPSettings.iClarity, 0);
				postf("New clarity threshold: % in map %\n", m.minVal, mMapMode);
			});
		})
	}

	getCycleThreshold {
		^mNumberBoxCycleThreshold.value;
	}

	setCycleThreshold { | tc |
		mNumberBoxCycleThreshold.valueAction_(tc);
	}

	mapMode_{ | mode |
		var gc;
		mMapMode = mode;
		switch (mode,
			iNormal,    { gc = Color.gray(0.5)},
			iClone,     { gc = Color.hsv(0.33, 0.8, 0.75)},
			iCloneFloat,     { gc = Color.hsv(0.33, 0.8, 0.75) },
			iSmooth,    { gc = Color.yellow(0.65)},
			iReference, { gc = Color.hsv(0.83, 0.8, 0.75)},
			iDiff,      { gc = Color.hsv(0.67, 0.8, 0.75)}
			// {  }
		);
		mGrid.gridColors_([gc, gc]);
		mGrid.fontColor_(gc);
		switch (mode,
			iDiff,   { this.setColorScale(mLayerSelected, mClusterSelected); mMapFileName = "" },
		);
		mbSized = true; // force a redraw?
	}

	toggleHorzGrid {
		mGridHorzGridSelectHz = mGridHorzGridSelectHz.not;
		if (Main.versionAtLeast(3,13), {
			if (mGridHorzGridSelectHz, {
				mGrid.horzGrid_( mGridLinesHz );
				mGrid.x.labelAppendString_(" Hz");
			}, {
				mGrid.horzGrid_( mGridLinesMIDI );
				mGrid.x.labelAppendString_(" MIDI");
				mGrid.numTicks_(nil,nil);
			});
			mUserViewMatrixBack !? { | uv | uv.onResize.value(uv) };
			mbRedrawBack = true;
		}, {
			mGrid.horzGrid_(if (mGridHorzGridSelectHz, { mGridLinesHz }, { mGridLinesMIDI } ));
		});
	}

	setLayer { | layer |
		mDropDownType.valueAction = layer;
	}

	// This method is called only on the NOW map and only if a TWIN map exists
	setTwinLayers { | twinLayers |
		mKeepNewest = (twinLayers[0] == mLayerSelected);
		// mKeepNewest = [mLayerSelected, VRPSettings.iClustersEGG, VRPSettings.iClustersPhon].includes(twinLayers[0]);
		// mKeepNewestBack = mKeepNewest and: (twinLayers[1] > 0);
		mKeepNewestBack = (twinLayers[1] > 0);
	}

	setMapHandler { | vMaps |
		this.addDependant(vMaps);
	}

	showingClusters {
		var bClusters;
		bClusters = (mLayerSelected == VRPSettings.iClustersEGG)
		or: (mLayerSelected == VRPSettings.iClustersPhon);
		^bClusters
	}

	mouseDownUp { arg inc;
		switch(inc,
		   -1, { nMouseDownRepeats = -1 },
			0, { nMouseDownRepeats = 0 },
			1, { nMouseDownRepeats = nMouseDownRepeats + 1 }
		);
	}

	init { | view, vrpCopy |
		var minFreq, maxFreq, gridLabelOffsetY, dBmax;

		mView = view;
		mHmargin = 0; mVmargin = 0;
		mLargerFont = Font.new("Arial", 9, bold: true, usePointSize: true);
		mGridFont   = Font.new(\Arial, 8.5, usePointSize: true);

		mLastPath = thisProcess.platform.recordingsDir;

		mCellCount = 0;
		mNewClusterOrder = nil;
		mVRPdata = nil;
		mVRPdataLoaded = nil;
		mClarityThreshold = 0.96;
		mbSized = true;
		mbRedrawBack = true;
		mbRedrawFront = true;
		mbRedrawTarget = true;
		mbRedrawCursor = true;
		mbClosing = false;
		mSavedMapPathStr = "";
		mMapFileName = "";
		mbSaveContext = false;
		mbMouseCellCoords = nil;
		mColorRect = Rect(0, 0, 120, 30);
		mColorbarText = "<metric>";
		myMetrics = VRPSettings.metrics.deepCopy;
		mLayerSelected = VRPSettings.iDensity;
		mKeepNewest = false;
		mKeepNewestBack = false;
		mSignalDSM = nil;
		stoppingTicks = 0;

		mClusterSelected = 0;
		mDictSelections = Dictionary.newFrom(
			[
				VRPSettings.iClustersEGG,  0,
				VRPSettings.iClustersPhon, 0
			]
		);

		mnClusters = 5;
		mDictNclusters 	= Dictionary.newFrom(
			[
				VRPSettings.iClustersEGG,  mnClusters,
				VRPSettings.iClustersPhon, mnClusters
			]
		);

		nMouseDownRepeats = 0;
		mbMouseDownShift = false;
		mCursorRectScaler = 1.0;
		ptMouseDownLast = Point(-10,-10); // dummy hidden point
		mGridHorzGridSelectHz = false;
		minFreq = VRPDataVRP.nMinMIDI;
		maxFreq = VRPDataVRP.nMaxMIDI;
		mGridSpecMIDI  = ControlSpec(minFreq, maxFreq, warp: \lin, units: "MIDI");
		mGridLinesMIDI = GridLines(mGridSpecMIDI);
		mGridSpecHz = ControlSpec(minFreq.midicps, maxFreq.midicps, warp: \exp, units: "Hz");
		mGridLinesHz = GridLines(mGridSpecHz);
		mGridSpecFreq = mGridSpecMIDI;
		this.setSPLscale();

		mGrid
		.smoothing_(false)
		.font_(mGridFont)
		.fontColor_(Color.gray(0.2));

		this.mapMode_(iNormal);
		this.setColorScale(VRPSettings.iDensity, mClusterSelected);

		mUserViewHolder = CompositeView(mView, mView.bounds);

		mUserViewMatrixBack = UserView(mUserViewHolder, mUserViewHolder.bounds);
		if (Main.versionAtLeast(3,13), {
			mUserViewMatrixBack.onResize_( { | uv |
				var w, xNumTicks, yNumTicks;
				if (mGridHorzGridSelectHz, {
					w = uv.bounds.width;
					case
					{ w < 210 } { xNumTicks = 1 }
					{ w < 500 } { xNumTicks = 3 }
					{ w < 800 } { xNumTicks = 4.2 }
					{ xNumTicks = 9 };
					mGrid.x.numTicks_(xNumTicks);
				});
				mGrid.y.numTicks_(uv.bounds.height / 40);
			})
		});

		mUserViewMatrixBack
		.background_(Color.white)
		.acceptsMouse_(true)
		.canFocus_(true)
		.addUniqueMethod(\getMyColor, { |view, ix, pos, offset=0 |
			mColorbarPalette.value(offset + mColorSpec.map(pos))
		});

		// ========= Mouse and key actions on maps ===============

		mUserViewMatrixBack.mouseDownAction_({ | uv, x, y, m, bn |
			if (bn == 0,
				{
					if (mColorRect.contains(x@y), {
						// Change the display layer, on left-click in the colour bar
						var dir = (x - mColorRect.center.x).sign;
						mDropDownType.valueAction = (mDropDownType.value + dir).mod(mDropDownType.items.size);
					}, {
						uv.focus(true);
						// Request playback, but only for NOW and TWIN and SMOOTH modes
						if ([iNormal, iClone, iSmooth].indexOf(mMapMode).notNil,
							{
								this.mouseDownUp(1);
								mbMouseCellCoords
								= mGridSpecMIDI.map(x / uv.bounds.width)
								@ mGridSpecSPL.map(1.0 - (y / uv.bounds.height));
								ptMouseDownLast = x@y;
								mbMouseDownShift = m.isShift;
						});
					});
			}); /* if bn == 0 */

			// Toggle lin/log grid, on right-click
			if (bn == 1,    {
				this.toggleHorzGrid;
			});
			this.invalidate(true);
		});

		mUserViewMatrixBack.mouseUpAction_({ | uv, x, y, m, bn |
			if (bn == 0, { 	this.mouseDownUp(-1) } );
		});

		// Ctrl-mousewheel to resize the listening rectangle
		mUserViewMatrixBack.mouseWheelAction_({ arg view, x, y, modifiers, xDelta, yDelta;
			if (modifiers.isCtrl, {
				mCursorRectScaler = mCursorRectScaler * exp(yDelta/360);
			});
		});

		// "Insert" key or '1' through '9' means request all values in the cell to that cluster
		mUserViewMatrixBack.keyDownAction_({ | v, c, mods, u, kCode, k |
			if (k == 0x01000006, { bRequestSetCentroid = 1 } );
			if (k.inclusivelyBetween(49, 57),
				{
					bRequestSetCentroid = k - 48;
				}
			);
		});

		// Allow dragging of _VRP.csv files onto the map window
		mUserViewMatrixBack.canReceiveDragHandler_({|v, x, y|
			var str, bOK = false;
			str = v.class.currentDrag;
			if (str.class == String, {
				if (VRPDataVRP.testSuffix(str), {
					bOK = true;
				})
			} );
			bOK
		})
		.receiveDragHandler_({|v, x, y|
			var str, bOK = false;
			str = v.class.currentDrag;
			this.loadVRPdataPath(str);
		});


		// ========= Drawing functions =================

		// How to draw the actual map data
		mUserViewMatrix = UserView(mUserViewHolder, mUserViewHolder.bounds);
		mUserViewMatrix
		.acceptsMouse_(false)
		.drawFunc_{ | uv |
			var b = uv.bounds.moveTo(0, 0).insetAll(0,0,1,1);
			Pen.use {
				// Flip the drawing of the matrix vertically, since the y-axis is flipped in the grid
				Pen.translate(0, b.height);
				Pen.scale(1, -1);
				if (mDrawableSparseMatrix.notNil, {
					if (uv.clearOnRefresh, {
						mDrawableSparseMatrix.invalidate;
					});
					mDrawableSparseMatrix.thresholdCount_(mNumberBoxCycleThreshold.value);
					mDrawableSparseMatrix.draw(uv, mKeepNewest);
					mbRedrawFront = false;
				});
			};
			uv.clearOnRefresh_(false);
		};

		this.initMenu();

		// How to draw the map grid, the colour bar,
		// and perhaps another map on the background
		mUserViewMatrixBack.drawFunc_{ | uv |
			var index = mLayerSelected;
			var b = uv.bounds.moveTo(0, 0).insetAll(0,0,1,1);
			if (uv.clearOnRefresh, {
				mGrid.bounds_(b);
				mGrid.draw;

				// If we are in "singer mode", draw a thicker line at 120 dB
				dBmax = VRPDataVRP.nMaxSPL;
				if (dBmax > 120, {
					var y = (1 - mGridSpecSPL.unmap(120)) * b.height;
					Pen.use {
						Pen.strokeColor = Color.gray(0.3);
						Pen.width_(2);
						Pen.moveTo(0@y);
						Pen.lineTo(b.width@y);
						Pen.stroke;
					}
				});

				// Draw the color scale bar over the grid
				mColorRect = Rect(
					50, // b.width * 0.08,
					20, // b.height * 0.05,
					180.max(100 * VRPMain.screenScale.x),
					35 * VRPMain.screenScale.y,
				);

				mLayerSelected = mDropDownType.value;  // should not be needed here, but...
				Pen.use {
					var nColors = mColorRect.width.half;
					var pos, rc, offs = 0;
					nColors.do ({ |k|
						rc = Rect(mColorRect.left+(k*2), mColorRect.top, 3, mColorRect.height) & mColorRect;
						pos = k / nColors.asFloat;
						if (this.showingClusters() and: (mClusterSelected == 0), {
							pos = (pos*mnClusters).floor/mnClusters;
							offs = -1;
						});
						Pen.color = uv.getMyColor(ix: index, pos: pos, offset: offs);
						Pen.addRect(rc);
						Pen.fill;
					})
				};

				// If the data is available, draw a histogram into the color bar
				if (mDrawableSparseMatrix.notNil , {
					var histArray = mDrawableSparseMatrix.histogram;
					if (histArray.isEmpty.not, {
						var n = histArray.size;
						var m = histArray.maxItem;
						var x = mColorRect.left;
						var y = mColorRect.bottom;
						Pen.use {
							Pen.smoothing_(true);
							// Pen.setShadow(1@1, 0.5, Color.gray);  // NYI
							Pen.width_(1.5);
							Pen.strokeColor_(Color.white);
							Pen.fillColor_(Color(0, 0, 0, 0.15));  // transparent black
							Pen.moveTo(x@y);
							histArray do: { | val, ix |
								y = val.linlin(0, m, mColorRect.bottom, mColorRect.top);
								Pen.lineTo(x@y);
								x = (ix+1).linlin(0, n-1, mColorRect.left+1, mColorRect.right-1);
								Pen.lineTo(x@y);
							};
							Pen.lineTo(x@mColorRect.bottom);
							Pen.lineTo(mColorRect.left@mColorRect.bottom);
							Pen.fillStroke;
						};
					});
				});

				// Draw the color-bar's grid and metric text
				Pen.use {
					var str;
					var mLayerRect = Rect(mColorRect.left, 1, mColorRect.width, mColorRect.top-2);
					Pen.strokeColor = Color.gray;
					Pen.strokeRect(mColorRect);
					mColorGrid
					.font_(mGridFont)
					.fontColor_(mColorGridFontColor)
					.gridColors_([Color.gray, Color.gray]);

					if (Main.versionAtLeast(3,13),
						{
							// ====== From SC v3.13.0 ========
							if (mColorSpec.warp.class != ExponentialWarp, { mColorGrid.numTicks_(5, nil) });
							if ((mLayerSelected < VRPSettings.iClustersEGG)
								or: (mClusterSelected > 0), {
									mColorGrid.x.labelAnchor_(\top)
									.labelAlign_(\center)
									.labelsShowUnits_(true)
									.labelOffset_(2 @ 2);
								} , {
									mColorGrid.x.labelAnchor_(\leftTop)
									.labelOffset_(4 @ 0);
							});
						}, {
							// ====== To SC v3.12.2 ========
							mColorGrid.x.labelOffset_(-10@2); // -14 * VRPMain.screenScale.y;
						}
					);

					mColorGrid.bounds_(mColorRect);
					mColorGrid.draw;
					str = mColorbarText;
					if (mClusterSelected > 0, { str = str.replace("#", mClusterSelected) });
					Pen.fillColor_(mUserViewMatrixBack.background);
					Pen.fillRect(mLayerRect);
					Pen.stringInRect(str, mLayerRect, mLargerFont, mColorGridFontColor, \bottom);
				};

				// If a file name for the map has been given, draw it discreetly at lower right
				if (mMapFileName.isEmpty.not, {
					Pen.use {
						var b, rc, str;
						str = mMapFileName;
						b = uv.bounds.moveTo(0, 0);
						rc = str.bounds(mGridFont);
						rc = rc.moveTo(b.width - 15 - rc.width, b.height - rc.height - 15);
						Pen.fillColor_(mUserViewMatrixBack.background);
						Pen.fillRect(rc);
						str.drawRightJustIn(rc, mGridFont, mGrid.x.fontColor);
					}
				});
				// Don't redraw all this background unnecessarily
				uv.clearOnRefresh_(false);
			});

			if (mDrawableSparseMatrixBack.notNil, {
				Pen.use {
					// Flip the drawing of the matrix vertically, since the y-axis is flipped in the grid
					Pen.translate(0, b.height);
					Pen.scale(1, -1);

					// Redraw all back-map cells when drawing
					mDrawableSparseMatrixBack.invalidate;

					if (this.showingClusters
						and: (mClusterSelected > 0)
						and: ( mMapMode != iDiff )
						and: { mVRPdata.notNil }, {
							var theLayer, pushPal;
							theLayer = mVRPdata.layers[\Density];

							// mDrawableSparseMatrixBack has been set up with the Density mapdata
							// On single-cluster maps, draw the density as a backdrop
							// Save any recolored density palette that might be in effect
							pushPal = theLayer.metric.palette;

							// Replace it with the default gray-only palette for the backdrop
							mDrawableSparseMatrixBack.setPalette(theLayer.metric.fnMyPalette);
							mDrawableSparseMatrixBack.thresholdCount_(mNumberBoxCycleThreshold.value);
							mDrawableSparseMatrixBack.draw(uv, mKeepNewestBack);

							// Restore the recolored density palette
							mDrawableSparseMatrixBack.setPalette(pushPal);

						}, {
							mDrawableSparseMatrixBack.thresholdCount_(1);
							mDrawableSparseMatrixBack.drawUnderlap(uv);
					});
				};
			});
			mbRedrawBack = false;

			// This is to ensure that the background is refreshed first
			if (mUserViewMatrix.notNil, { mUserViewMatrix.clearOnRefresh_(true) });
		} /* mUserViewMatrixBack.drawFunc */;

		this.invalidate(true);

		// How to draw the cursor and/or the region selected in the signal
		mUserViewCursor = UserView(mUserViewHolder, mUserViewHolder.bounds);
		mUserViewCursor
		.acceptsMouse_(false)
		.clearOnRefresh_(true)
		.drawFunc_{ | uv |
			if ( mCursorRect.notNil, {
				Pen.use {
					Pen.fillColor = mCursorColor;
					Pen.strokeColor = Color.black;
					Pen.fillRect(mCursorRect);
					Pen.strokeRect(mCursorRect);
				};
				// mbRedrawCursor = false;
			});
		};

		// How to draw the target region, if any
		mUserViewTarget = UserView(mUserViewHolder, mUserViewHolder.bounds);
		mUserViewTarget
		.acceptsMouse_(false)
		.clearOnRefresh_(true)
		.drawFunc_{ | uv |
			if (uv.clearOnRefresh, {
				if ( mSignalDSM.notNil and: ([iNormal, iClone, iCloneFloat].includes(mMapMode)), {
					var b = uv.bounds.moveTo(0, 0).insetAll(0,0,1,1);
					Pen.use {
						// Flip the drawing of the matrix vertically,
						// since the y-axis is flipped in the grid
						Pen.translate(0, b.height);
						Pen.scale(1, -1);
						mSignalDSM.drawUnderlap(uv);  // draw as a hash pattern
					};
					uv.clearOnRefresh_(false)
				};
				);
			});
			mbRedrawTarget = false;
		};

		// ======== Layouts and resizing =============

		fnFixedAspectLayout = { | v |
			var cell, aspect;
			cell = Size(
				v.bounds.width  / VRPDataVRP.vrpWidth,
				v.bounds.height / VRPDataVRP.vrpHeight
			);
			aspect = cell.width / cell.height;
			if (aspect > 2.0,
				{
					mHmargin = (v.bounds.width  - (v.bounds.width  * 2.0 / aspect)) / 2.0 ;
					mVmargin = 0;
				}, {
					mVmargin = (v.bounds.height - (v.bounds.height * aspect / 2.0)) / 2.0 ;
					mHmargin = 0;
				}
			);
		};

		mView.onResize_( { | v |
			if (v.bounds.notNil, {
				mStaticTextCycleThreshold.visible_(v.bounds.width > 400);
				mStaticTextInfo.visible_(v.bounds.width > 400);
			}
			);
			if (VRPDataVRP.bFixedAspectRatio, {
				fnFixedAspectLayout.(mUserViewHolder)
			});

			// One m*margin only can be nonzero - when the cell aspect ratio is fixed
			mUserViewHolder.layout_(
				VLayout(
					mVmargin,
					[
						HLayout(
							mHmargin,
							[
								StackLayout(
									mUserViewCursor,
									mUserViewTarget,
									mUserViewMatrix,
									mUserViewMatrixBack
								).mode_(\stackAll)
								, stretch: 50 // Force the menu to take up as little space as possible!
							],
							mHmargin
						)
					],
					mVmargin
				).margins_(0);
			);
			mbSized = true;
		} );

		mView.layout_(
			VLayout(
				[
					HLayout(
						[mStaticTextCycleThreshold, stretch: 1],
						[mNumberBoxCycleThreshold, stretch: 1],
						[10, stretch: 10],
						[mButtonLoadVRPdata, stretch: 1],
						[mButtonSaveVRPdata, stretch: 1],
						[mButtonSaveVRPimage, stretch: 1],
						[nil, stretch: 2]   // Force the controls to take up as little space as possible
					), stretch: 0
				],

				[
					HLayout(
						[mStaticTextLayer, stretch: 1],
						[mDropDownType, stretch: 10],
						[mStaticTextCluster, stretch: 1],
						[mSliderCluster, stretch: 10],
						[mStaticTextInfo, stretch: 1]
					), stretch: 0
				],

				[
					mUserViewHolder, stretch: 10
				]
			)
		);
		mView.layout.margins_(5);

		if (vrpCopy.notNil, {
			var layer, sliderPos, fileStr;
			mVRPdata = vrpCopy.mVRPdata.deepCopy;
			#layer, mnClusters, mClusterSelected, fileStr, mGridHorzGridSelectHz = vrpCopy.mapSwitches;
			this.setSPLscale;
			mDropDownType.valueAction_(layer);
			sliderPos = (mClusterSelected/mnClusters).round(0.01);
			mSliderCluster.value_(sliderPos);
			mMapFileName = fileStr;
		});

		this.updateView;
		mView.onClose_({this.close});
		AppClock.sched(0.5, { mbSized = true }); // Schedule an extra refresh
	} /* init */

	initMenu {
		var static_font = Font.new(\Arial, 8, usePointSize: true);

		mStaticTextCycleThreshold = StaticText(mView, Rect())
		.string_("Cycle threshold:")
		.font_(static_font);
		mStaticTextCycleThreshold
		.fixedWidth_(mStaticTextCycleThreshold.sizeHint.width)
		.maxHeight_(20)
		.stringColor_(Color.white);

		mNumberBoxCycleThreshold = NumberBox(mView, Rect())
		.clipLo_(1)
		.step_(1)
		.scroll_step_(5)
		.align_(\center)
		.fixedWidth_(28)
		.action_({ mbSized = true })
		.value_(1);

		mButtonLoadVRPdata = Button(mView, Rect());
		mButtonLoadVRPdata
		.states_([["Load Map"]])
		.action_( { |btn|
			this.loadVRPdataDialog( { arg n;
				mnClusters = n;
			} );
			mbSized = true;
		})
		.canReceiveDragHandler_({|v, x, y|
			var str, bOK = false;
			str = v.class.currentDrag;
			if (str.class == String, {
				if (VRPDataVRP.testSuffix(str), {
					bOK = true;
				} , {
					format("Filename issue: %", PathName(str).fileName).warn;
				})
			} );
			bOK
		})
		.fixedWidth_(60)
		.receiveDragHandler_({|v, x, y|
			var str, bOK = false;
			str = v.class.currentDrag;
			this.loadVRPdataPath(str);
		})
		.enabled_(true);

		mButtonSaveVRPdata = Button(mView, Rect());
		mButtonSaveVRPdata
		.states_([["Save Map"]])
		.action_( { |btn|
			if (mVRPdata.notNil, {
				this.saveVRPdataDialog;
			}
			);
		})
		.fixedWidth_(60)
		.enabled_(false);

		mButtonSaveVRPimage = Button(mView, Rect());
		mButtonSaveVRPimage
		.states_([["Save Image"]])
		.action_( { |btn|
			var rect = (mDropDownType.bounds union: mUserViewHolder.bounds).insetBy(-5);
			this.writeImage(mView, rect, mLastPath, { | retPath | mLastPath = retPath });
		})
		.fixedWidth_(64)
		.canReceiveDragHandler_({ |v| v.class.prClearCurrentDrag; }); // prevent drop

		mStaticTextLayer = StaticText(mView, Rect())
		.string_("Layer:")
		.font_(static_font);
		mStaticTextLayer
		.fixedWidth_(mStaticTextLayer.sizeHint.width)
		.stringColor_(Color.white);

		mDropDownType = PopUpMenu(mView, [0, 0, 100, 30]);
		mDropDownType
		.items_( myMetrics collect: { | m | m.menuText } )
		.action_({ | v |
			if (v != mLayerSelected, {
				mLayerSelected = v.value;
				mDictNclusters[mLayerSelected]
				!? { |n| mnClusters = n } ;
				mDictSelections[mLayerSelected]
				!? 	{ |s| 		// if showing clusters
					mSliderCluster.step_(1 / mnClusters);
					mSliderCluster.valueAction_(s.value/mnClusters);
				}
				?? 	{
					this.setColorScale(v.value, mClusterSelected);		// if showing something else
				};
				this.changed(this, \selectLayer, v.value);
				mbSized = true;
				this.invalidate(true);
				this.updateView();
			});
		})
		.allowsReselection_(true)		// simplifies forced redraws
		.resize_(4);

		mStaticTextCluster = TextField(mView, [0, 0, 50, 30])
		.font_(static_font)
		.string_("All");
		mStaticTextCluster
		.maxWidth_(25)
		.align_(\center)
		.enabled_(false);

		mnClusters = 5;
		mClusterSelected = 0;
		mSliderCluster = Slider(mView, [0, 0, mView.bounds.width*0.2, 30]);
		mSliderCluster
		.maxHeight_(24)
		.maxWidth_(50)
		.value_(0.0)
		.orientation_(\horizontal)
		.resize_(5);
		mSliderCluster.action_{ |s|
			s.step_(1 / mnClusters);
			mClusterSelected = (s.value * mnClusters).round(1).asInteger;
			mDictSelections.add(mLayerSelected -> mClusterSelected);
			this.setColorScale(mLayerSelected, mClusterSelected);
			mbSized = true;
			this.invalidate(true);
		};

		// Ignore default keys if Alt is pressed
		mSliderCluster.keyDownAction = { |v, char, mod, unicode, keycode, key|
			var ret_val = nil;
			if (mod.isAlt, { ret_val = false },
				{ ret_val = v.defaultKeyDownAction(char, mod, unicode, keycode, key)}
			);
			ret_val
		};

		mStaticTextInfo = StaticText(mView, Rect(0, 0, 200, 30))
		.font_(Font.new(\Arial, 8, true, false, true))
		.string_("XXXXXXXXXXXXXXXXXXXXX");
		mStaticTextInfo
		.fixedWidth_(mStaticTextInfo.minSizeHint.width)
		.fixedHeight_(16 * VRPMain.screenScale.y)
		.align_(\center);

	} /* initMenu */

	update { | menu, who, what, newValue |
		switch (what,
			\selectCluster,
			// newValue is in the range 0..nClusters
			{ 	mDictSelections.at(who.metricID)
				!? { mDictSelections.add(who.metricID -> newValue) };
				mbSized = true;  // invokes a redraw
				// Check if the displayed cluster type matches the changed one
				if (who.metricID == mLayerSelected, { mSliderCluster.valueAction_(newValue/mnClusters) } );
			},

			\reorderClusters,
			{
				// Should affect only the NOW map, even if more are on display
				if (mMapMode == iNormal, {
					mDropDownType.valueAction = who.metricID;
					mNewClusterOrder = newValue;
					this.invalidate(false);
				});
			},

			\numberOfClusters,
			{
				// If a number of clusters is changed from its panel,
				// reallocate the corresponding cluster map, thus clearing it
				var oldNumber = mDictNclusters.at(who.metricID);
				if (oldNumber.notNil
					and: (oldNumber != newValue),	// If not changed, leave it
					// and: (mbSized.not), 			// Only if no draw is pending
					{
						if (mVRPdata.notNil,
							{
								mDictNclusters.add(who.metricID -> newValue);
								mVRPdata.initClusteredLayers(
									VRPDataVRP.vrpHeight+1,
									VRPDataVRP.vrpWidth+1,
									who.metricID,
									newValue,
									(mMapMode == iDiff)
								);
								if (mLayerSelected == who.metricID, {
									mClusterSelected = 0;
									this.setColorScale(mLayerSelected, mClusterSelected);
									mbSized = true;  // invokes a redraw
								});
							}, {
								format("debug: mVRPdata=%, newValue=%", mVRPdata, newValue).warn;
							}
						)
				})
			},

			\dialogSettings,
			{
				this.stash(newValue);
			},

			\splRangeChanged,
			{
				VRPDataVRP.configureSPLrange(newValue);
				this.setSPLscale();
			},

			\mapWasDeleted,
			{	// If a deleted map was a difference map,
				// then clear also the underlap map that we have here.
				if (newValue.mapMode == VRPViewVRP.iDiff, {
					mVRPdata.initUnderlap;
				})
			},

			\newMapWasLoaded,
			{	// If this is a TWIN map, and the new map was the NOW map,
				// update the display (map and filename)
				if ((mMapMode == iClone) or: (mMapMode == iCloneFloat) and: (who.mapMode == iNormal), {
					mMapFileName = newValue;
					mbSized = true;
				})
			},

			\targetOverlay,
			{
				if ([iNormal, iClone, iCloneFloat, iSmooth].includes(mMapMode), {
					mSignalDSM = newValue;
					mbSized = true;
				})
			},

			// else
			{ warn("Unknown change notification") }
		);
	} /* .update */

	updateView { | data=nil |
		var is_clusters;
		var infoStr, infoStrBrightness;
		var ixM, nPlaying;

		is_clusters = this.showingClusters();

		// Show or hide various controls
		mButtonLoadVRPdata.visible = [iNormal, iReference].indexOf(mMapMode).notNil;
		mButtonSaveVRPdata.visible = [iNormal, iDiff, iSmooth].indexOf(mMapMode).notNil;
		mStaticTextLayer.visible_(true);
		mSliderCluster.visible_(is_clusters);

		// Update the cluster text
		mStaticTextCluster.string_(
			if (mClusterSelected == 0, "All", mClusterSelected.asString )
		);
		mStaticTextCluster.visible_(is_clusters);

		// Update the info text for the current layer
		ixM = mLayerSelected;
		if (data.isNil, {
			if (mMapMode == iReference, {
				var m;
				m = mVRPdata.layers[\Clarity].metric;
				m.minVal = mClarityThreshold;
				m.setTrendText(mStaticTextInfo);
			});
		}, {
			var m, sym = myMetrics[ixM].class.symbol;
			m = data.layers[\Clarity].metric(mClusterSelected);
			m.minVal_(mClarityThreshold);
			data.layers[sym].metric(mClusterSelected).setTrendText(mStaticTextInfo);
		});

		// Improve its legibility by compensating bkgnd for textcolor
		infoStrBrightness = ((mStaticTextInfo.stringColor.asArray[0..2])*[1,1.5,0.7]).sum/3;
		mStaticTextInfo.background_(Color.gray((infoStrBrightness + 0.5).mod(1)));

		mGrid.font_(mGridFont);
	} /* updateView{} */

	stash { | settings |
		var inPath, tmp;
		tmp = settings.vrp.clarityThreshold;
		if (tmp != mClarityThreshold,
			{
				this.setClarityThreshold(tmp);
				AppClock.sched(0.2, { mDropDownType.valueAction = VRPSettings.iClarity });
			}
		);

		if (mGridHorzGridSelectHz != settings.vrp.bHzGrid, {
			this.toggleHorzGrid;
		});

		// Set on LOAD _VRP in script
		tmp = settings.vrp.loadedVRPdata;
		if ((mMapMode == iNormal) and: { tmp.notNil },
			{
				mMapFileName = tmp.mLastPathName.fileName;
				mVRPdataLoaded = tmp;
				// Must wait a little for the redraw to catch up:
				AppClock.sched(1.0, { settings.vrp.loadedVRPdata = nil });
			}
		);
	}

	fetch { | settings |

		if (settings.waitingForStash, {
			this.stash(settings);
		});

		settings.vrp.clarityThreshold = mClarityThreshold;
		settings.vrp.bHzGrid = mGridHorzGridSelectHz;

		// Flag a request for picking a centroid from a map
		if (bRequestSetCentroid > 0, {
			this.stashForCentroid(settings, bRequestSetCentroid);
			bRequestSetCentroid = 0;
		});

		// This serves to remove green text
		// from the Save Map Button, if things have changed
		if (settings.vrp.wantsContextSave and: (settings.checkMapContext.not), {
			mbSaveContext = false;
		});
	}

	stashForCentroid { arg settings, cNum;
		var cps = settings.clusterPhon;
		var metrics = cps.clusterMetrics;
		var cValues = 0.0 ! (metrics.size);
		var idx_midi = VRPDataVRP.frequencyToIndex( mbMouseCellCoords.x );
		var idx_spl  = VRPDataVRP.amplitudeToIndex( mbMouseCellCoords.y );
		var dsm;
		if (mVRPdata.notNil and: (cps.mapCentroid.isNil), {
			metrics do: { arg sym, i;
				dsm = mVRPdata.layers[sym].mapData;	// Find the right DrawableSparseMatrix
				cValues[i] = dsm.at(idx_spl, idx_midi); // Get the value
			};
			if (cValues[0].notNil, { 			// Only if the cell was not empty
				cps.mapCentroid = cValues;
				cps.mapCentroidNumber = cNum;
			});
		}, {
			"mVRPdata is nil".warn;
		})
	} /* .stashForCentroid */

	updateData { | data |
		var vrpd;
		var dsg;
		var idx_midi, idx_spl, cellValue;
		var layerSym;

		if (mbClosing, { ^nil } );

		// ds = data.settings;
		dsg = data.settings.general;
		if (dsg.guiChanged, {
			mView.background_(dsg.getThemeColor(\backPanel));
			mStaticTextCycleThreshold.stringColor_(dsg.getThemeColor(\panelText));
			mStaticTextLayer.stringColor_(dsg.getThemeColor(\panelText));
			mUserViewMatrixBack.background_(dsg.getThemeColor(\backMap));
			mColorGridFontColor = dsg.getThemeColor(\dullText);
			mbSized = true;
		});

		if (mNewClusterOrder.notNil, {
			data.vrp.reorder(mNewClusterOrder, mLayerSelected);
			mNewClusterOrder = nil;
		} );

		if (mVRPdataLoaded.notNil,
			{
				if (mMapMode == iReference, {
					mVRPdata = mVRPdataLoaded;
				} , {
					data.vrp.reset(mVRPdataLoaded);
				});

				this.setColorScale(mLayerSelected, mClusterSelected);
				mVRPdataLoaded = nil;
				mbSized = true;

				// Let the TWIN map, if any, know that the file name changed
				this.changed(this, \newMapWasLoaded, mMapFileName);
			}
		);

		if (mMapMode >= iReference, {
			vrpd = mVRPdata;
		} , {
			vrpd = data.vrp;	// NOW and TWIN map views are updated through "data"
		});

		this.updateView(vrpd);

		mVRPdata = vrpd; // Remember for saving

		if (data.general.stopping, {
			if (data.settings.vrp.wantsContextSave and: (data.io.eof), {
				// set Save Map button text to green if context-script will be written
				if (mbSaveContext = data.settings.checkMapContext, {
					mButtonSaveVRPdata.states_([["Save Map", Color.green(0.6)]]);
				});
			});
			mButtonLoadVRPdata.enabled = true;
			mButtonSaveVRPdata.enabled = true;
			stoppingTicks = stoppingTicks + 1;
		}); // Enable if stopping

		if (data.general.starting, {
			mMapFileName = "";						// Loaded map is no longer valid
			mButtonLoadVRPdata.enabled = false; 	// Disable when starting
			mButtonSaveVRPdata.enabled = false;
			mCellCount = 0;
			mbRedrawBack = true;					// Refresh the back to clear any histogram
			this.invalidate(true);					// Clear the old graph
			mbRedrawTarget = true;
		});

		if (data.general.started, {
			mbRedrawFront = true;					// Refresh the map on every frame
		} );

		// Update the graph depending on the type selected in the dropdown menu
		// The refDensity matrix is used by most layers for calculating the per-cell means
		mDrawableSparseMatrixBack = nil;
		mDrawableSparseMatrix.notNil.if { mDrawableSparseMatrix.mbActive_(false) };

		// Set up the graphic elements for the currently selected layer
		vrpd.layers[\Clarity].metric.minVal_(mClarityThreshold);
		layerSym = VRPSettings.metrics[mLayerSelected].class.symbol;	// Temporary
		theLayer = vrpd.layers[layerSym];
		theMetric = theLayer.metric(mClusterSelected);
		mDrawableSparseMatrix = theLayer.mapData(mClusterSelected);

		// Test: set the colorbar text from the phonclusters
		if ((layerSym == \ClustersPhon) and: (mClusterSelected > 0), {
			var strParts = theMetric.colorBarText.split($#);
			var labelStr = strParts[0] ++ "#: " ++ data.settings.clusterPhon.clusterLabels[mClusterSelected-1];
			if (theMetric.colorBarText != labelStr, {
				theMetric.colorBarText = labelStr;
				mbSized = true;  // redraw if changed
			});
		});

		// Disallow cell thresholding on the Clarity layer
		// - it can contain cells that other layers don't have.
		if (layerSym == \Clarity, {
			mDrawableSparseMatrix.refDensity_(nil);
			// theMetric.minVal = mClarityThreshold;
		}, {
			mDrawableSparseMatrix.refDensity_(vrpd.layers[\Density].mapData);
		});
		mColorbarPalette = theMetric.getPaletteFunc;
		mColorbarText = theMetric.colorBarText;

		case
		{ (layerSym == \ClustersEGG) or: (layerSym == \ClustersPhon) }
		{
			mnClusters = theLayer.cCount;
			mClusterSelected = mDictSelections[mLayerSelected];
			if (mClusterSelected > 0, {
				mDrawableSparseMatrixBack = vrpd.layers[\Density].mapData;
				mDrawableSparseMatrixBack.mbActive_(true);
				if (mMapMode == iDiff, {
					mDrawableSparseMatrixBack.setPalette(VRPMetric.fnPrePostPalette);
				});
				if (data.general.started, {
					mbRedrawBack = true;					// Refresh the map on every frame
				} );
			});
			mCursorPalette = theLayer.metric(0).palette;
		};

		if (mDrawableSparseMatrix.notNil, {
			mDrawableSparseMatrix.mbActive_(true);
		});

		mDrawableSparseMatrixBack.isNil.if {
			// The .underlapBack (a DrawableSparseMatrix)
			// will be valid but empty unless a DIFF map exists...
			mDrawableSparseMatrixBack = mVRPdata.underlapBack;
		};

		if (mDrawableSparseMatrixBack.notNil, {
			// ...so this is almost always true, actually
			mDrawableSparseMatrixBack.mbActive_(true)
		});

		// Compute the selected metric's distribution, but only if we are not analyzing
		if (mDrawableSparseMatrix.notNil
			and: { data.general.started.not }
			and: { mDrawableSparseMatrix.histogram.isEmpty },
			{
				var nBins, i = mLayerSelected;
				nBins = ((mLayerSelected < VRPSettings.iClustersEGG) or: { mClusterSelected > 0 })
				.if { mColorRect.width.half } { mnClusters+1 } ;
				mDrawableSparseMatrix.makeHistogram(
					nBins, theMetric.minVal, theMetric.maxVal, theMetric.colorBarWarpType
				);
				// Flag for a redraw iff a histogram was built
				if (mDrawableSparseMatrix.histogram.isEmpty.not, { mbRedrawBack = true });
			}
		);

		// Update the cursor
		if (vrpd.currentAmplitude.notNil and: vrpd.currentFrequency.notNil and: vrpd.currentClarity.notNil, {
			var idx_midi = VRPDataVRP.frequencyToIndex( vrpd.currentFrequency );
			var idx_spl = VRPDataVRP.amplitudeToIndex( vrpd.currentAmplitude );
			var px = VRPDataVRP.frequencyToIndex( vrpd.currentFrequency, mUserViewCursor.bounds.width );
			var py = mUserViewCursor.bounds.height - 1 - // Flip vertically
			VRPDataVRP.amplitudeToIndex( vrpd.currentAmplitude, mUserViewCursor.bounds.height );

			mCursorRect = Rect.aboutPoint(
				px@py,
				iCursorWidth.half.asInteger,
				iCursorHeight.half.asInteger
			);

			// Update the cursor color depending on the type selected in the dropdown menu
			switch (mLayerSelected,
				VRPSettings.iClustersEGG,
				{
					mCursorColor = mCursorPalette.(vrpd.currentCluster ? 0);
				},
				VRPSettings.iClustersPhon,
				{
					mCursorColor = mCursorPalette.(vrpd.currentClusterPhon ? 0)
				},
				{mCursorColor = Color.clear }
			);
			mbRedrawCursor = true;
		});

		if ( data.general.started.not,
			{
				// Invoke the map-listening mechanism?
				var target = VRPDataPlayer.iEmptyCell; // Assume cell is empty: ignore
				var wCur = 5, hCur = 5;				// Dummy non-nil values

				mCursorRectScaler = data.player.requestScaling(mCursorRectScaler);
				if (nMouseDownRepeats == -1, {		// Mouse has just been released
					data.player.markMouseUp();
					this.mouseDownUp(0);
					/*nMouseDownRepeats = 0; */
				});

				// If applicable, register the clicked place for listening
				if ((nMouseDownRepeats == 1) and: mDrawableSparseMatrix.notNil, {
					// Get the content of the clicked cell
					var idx_midi = VRPDataVRP.frequencyToIndex( mbMouseCellCoords.x );
					var idx_spl  = VRPDataVRP.amplitudeToIndex( mbMouseCellCoords.y - VRPDataVRP.nMaxSPL );
					cellValue = mDrawableSparseMatrix.at(idx_spl, idx_midi);
					this.mouseDownUp(-1);
					// Determine whether or not this search is cluster-specific
					if (cellValue.notNil, {
						target = VRPDataPlayer.iAnyCell;  // Match cells, ignore clusters
						if (this.showingClusters(), {
							if (mClusterSelected == 0, {
								// Get the number of the dominant cluster in the cell
								target = cellValue[0];
							}, {
								// Get the cluster number of the active layer
								target = mClusterSelected - 1;
							});
						});
					});

					// Signal a request for listening
					data.player.markForListening(mbMouseCellCoords,	target,	mLayerSelected,	mbMouseDownShift);
				});

				// Update the map-player cursor
				if ((data.player.target[1] != VRPDataPlayer.iEmptyCell)
					// Don't display region on BEFORE or DIFF maps
					and: [iReference, iDiff].indexOf(mMapMode).isNil
					// and: VRPDataPlayer.enabled,
					and: (data.player.available > 1),
					{
						// Set up a cursor rectangle that bounds the region to be searched for
						// The position is stored in data.player so that both NOW and TWIN maps can update it
						var ptCell = data.player.target[0], pt=Point();
						var alpha, uv;
						uv = mUserViewMatrixBack;
						wCur = data.player.midiTolerance * uv.bounds.width / VRPDataVRP.vrpWidth;
						hCur = data.player.levelTolerance * uv.bounds.height / VRPDataVRP.vrpHeight;
						pt.x = mGridSpecMIDI.unmap(ptCell.x) * uv.bounds.width;
						pt.y = (1.0 - mGridSpecSPL.unmap(ptCell.y+VRPDataVRP.nMaxSPL)) * uv.bounds.height;
						mCursorRect = Rect.aboutPoint(pt, wCur, hCur);
						alpha = 1.0 - data.player.representativity().sqrt;
						mCursorColor = Color.new(0, 0, 0, alpha);
					}, {
						mCursorRect = nil;
					}
				);

				// If present, display a target map or a selected-signal map
				// in the cursor view, i.e., on top of the regular map
				if (mSignalDSM.notNil, {
					mSignalDSM.mbActive_(true);
					mSignalDSM.invalidate;
				});

				// If a context for this map is unambiguous, save it as a script
				if (mbSaveContext and: (mSavedMapPathStr.isEmpty.not), {
					this.saveContextScript(data);
					mSavedMapPathStr = "";
				});
				if (mbSaveContext.not, {
					mButtonSaveVRPdata.states_([["Save Map"]]);
				});

				// Invoke an extra redraw so as to draw the new histogram when stopped
				if (stoppingTicks > 0, {
					mbSized = true;
					stoppingTicks = 0;
				});
		});

		if (mbSized, {
			mbRedrawBack  = true;
			mbRedrawFront = true;
			mbRedrawTarget = true;
			mbRedrawCursor = true;
			this.invalidate(true);
			mbSized = false;
		});

		if (mbRedrawBack, {
			mUserViewMatrixBack.refresh;
		});

		if (mbRedrawFront, {
			mUserViewMatrix.refresh;
		});

		if (mUserViewTarget.visible and: mbRedrawTarget, {
			mUserViewTarget.refresh;
		});

		if (mbRedrawCursor, {
			mUserViewCursor.refresh;
		});

	} /* .updateData */

	saveContextScript { arg data;
		var contextStr = data.mapContextString(mSavedMapPathStr);
		var pn = PathName(mSavedMapPathStr);
		var n = pn.pathOnly +/+ pn.fileName ++ ".Context.txt";
		var f = File(n.standardizePath, "w");
		f.write(contextStr);
		f.close;
		("Saved" + n).postln;
		mbSaveContext = false;
	}

	buildTargetDSM {
		var palette = { | val | Color.magenta(0.5) };
		var dsm = mVRPdata.layers[\Density].mapData.deepCopy;
		dsm.setPalette(palette);
		dsm.refDensity = dsm;
		dsm.thresholdCount = mNumberBoxCycleThreshold.value.asInteger;
		^dsm;
	}

	checkClusterCounts { arg dataReference, dataTarget, strContext;
		// Check the numbers of clusters
		// and return false if they do not match properly
		var bDoClusters = true;
		var nCrefE = dataReference.layers[\ClustersEGG].cCount;
		var nCtarE = dataTarget.layers[\ClustersEGG].cCount;
		var nCrefP = dataReference.layers[\ClustersPhon].cCount;
		var nCtarP = dataTarget.layers[\ClustersPhon].cCount;
		var nCpanE = mVRPdata.layers[\ClustersEGG].cCount;
		var nCpanP = mVRPdata.layers[\ClustersPhon].cCount;

		if ((nCrefE != nCtarE)
			or: (nCrefP != nCtarP)
			or: (nCpanE != nCrefE)
			or: (nCpanP != nCrefP)
			, {
				format("Cluster counts mismatch - %:\n\t   EGG\tPhonTypes", strContext).warn;
				format("Panel\t%\t%", nCpanE, nCpanP).postln;
				format("Before\t%\t%", nCrefE, nCrefP).postln;
				format("Now\t\t%\t%", nCtarE, nCtarP).postln;
				bDoClusters = false;
			}
		);
		^bDoClusters
	}

	// A call to this method is requested only once, in VRPViewMaps.addMap(3)
	computeDiffs { arg dataReference, dataTarget;
		var doClusters = this.checkClusterCounts(dataReference, dataTarget, "skipping layers");

		VRPSettings.metrics.do { | m, ix |
			var sym = m.class.symbol;
			var layer = mVRPdata.layers[sym];

			if (layer.class == VRPDataClusterMap, {
				if (doClusters, {
					layer.setDiffs (dataTarget.layers[sym], dataReference.layers[sym]);
				});
			}, {
				switch (sym )
				{ \Density } { layer.mapData.setRatios(dataTarget.layers[sym].mapData, dataReference.layers[sym].mapData) }
				{ \Clarity } { layer.mapData.setMins(dataTarget.layers[sym].mapData, dataReference.layers[sym].mapData) }
				{ \Qdelta  } { layer.mapData.setRatios(dataTarget.layers[sym].mapData, dataReference.layers[sym].mapData) }
				/* else */   { layer.mapData.setDiffs(dataTarget.layers[sym].mapData, dataReference.layers[sym].mapData) };
			}
			);
		};

		// Find which cells are present in only one of the maps
		// Cells only in dataReference get the value -1
		// Cells only in dataTarget    get the value +1
		mVRPdata.underlapBack.mapUnderlap(
			dataTarget.layers[\Density].mapData,
			dataReference.layers[\Density].mapData
		);

		// Let the two maps both know their common underlap region
		dataTarget.underlapBack = mVRPdata.underlapBack;
		dataReference.underlapBack = mVRPdata.underlapBack;

		// Force an update of color scales etc
		mDropDownType.valueAction = mDropDownType.value;
		mButtonSaveVRPdata.enabled = true;
		mbSized = true;
	} /* .computeDiffs */

	// Calls to this method are made in VRPViewMaps.addMap(4)
	interpolateSmooth { arg dataSource;
		var bOK = this.checkClusterCounts(mVRPdata, dataSource, "can't smooth");
		if (bOK.not, {
			^false
		});
		mVRPdata.interpolateSmooth(dataSource);
		// Force an update of color scales etc
		mDropDownType.valueAction = mDropDownType.value;
		mButtonSaveVRPdata.enabled = true;
		mVRPdata.mLastPathName_(PathName(""));
		mMapFileName = "";
		^true
	} /* .interpolateSmooth */

	loadVRPdataPath { arg inPath;
		var tempVRPdata, nClusters;
		tempVRPdata = VRPDataVRP.new(nil);
		nClusters = tempVRPdata.loadVRPdata(inPath);
		if (nClusters < 0, {
			mVRPdataLoaded = nil;
			format("Load failed of %, code=%", PathName.new(inPath).fileName, nClusters).error;
		},{
			mVRPdataLoaded = tempVRPdata;
			mLastPath = PathName.new(inPath).pathOnly;
			mMapFileName = tempVRPdata.mLastPathName.fileName;
			this.setSPLscale();
		});
		^nClusters
	}

	loadVRPdataDialog { | funcDone |
		var nClusters;
		VRPMain.openPanelPauseGUI({ | path |
			nClusters = this.loadVRPdataPath(path);
		}, { funcDone.value(nClusters) }, path: mLastPath);
	}  /* .loadVRPdataDialog{} */

	saveVRPdataDialog {
		var suff = "_VRP.csv";
		switch (mMapMode,
			iSmooth, { suff = "_S_VRP.csv" },
			iDiff,   { suff = "_D_VRP.csv" }
		);
		VRPMain.savePanelPauseGUI({ | path |
			mVRPdata.saveVRPdata(path, mMapMode );
			mSavedMapPathStr = path;
			mMapFileName = mVRPdata.mLastPathName.fileName;
			mLastPath = PathName.new(path).pathOnly;
			mbRedrawBack = true; // to update the displayed file name
		}, path: mLastPath, wantedSuffix: suff);
	}

	writeImage { arg view, rect, startPath, fnReturnPath;
		var currentPath = startPath;
		var iTotal = Image.fromWindow(view, rect);
		var tmpWnd = iTotal.plot(
			name: "Press F to save this image to a file, L to list all formats",
			bounds: rect.moveTo(200,200),
			freeOnClose:false,
			showInfo: false);

		var saveFunc = { arg startHere;
			var lastPath = startHere;
			Dialog.savePanel({ arg path;
				var pn = PathName.new(path);
				var p = path;
				var ext = pn.extension;
				var name = pn.fileNameWithoutExtension;
				var supported = false;

				lastPath = pn.pathOnly;
				Image.formats do: { | str |	if (str == ext, { supported = true } ) };

				if (supported.not, {
					format(("Unsupported format % - using" + ".png".quote), ext.quote).warn;
					p = lastPath +/+ name ++ ".png";
				});

				iTotal.write(p, format: nil);
				("Image saved to" + p).postln;
				fnReturnPath.(lastPath);
				currentPath = lastPath;
			}, path: startHere);
		};

		tmpWnd.view.keyDownAction_({ arg v, c, mods, u, kCode, k;
			var bHandled = true;
			case
			{ c.toLower == $f } { saveFunc.(currentPath) }		// F for file save
			{ c.toLower == $l } { format("Supported image file formats:\n % ", Image.formats).postln }
			{ c.toLower == $c } { Image.closeAllPlotWindows; iTotal.free }
			{ k == 0x01000000 } { tmpWnd.close(); iTotal.free }	// ESC for skipping
			{ bHandled = false } ;
			bHandled
		});

		tmpWnd
		.setInnerExtent(rect.width, rect.height)
		.front;
	} /* .writeImage */

	close {
		mbClosing = true;
		this.release;
		// mVRPdata !? { _.free };
	}

	toggleMap { | mode |
		// when supporting only a single map view
		"Multiple maps are not supported.".warn;
	}

	toggleLayout {
		// when supporting only a single map view
		"Multiple maps are not supported.".warn;
	}

	mapExists { | mode |
		"Multiple maps are not supported.".warn;
		^false
	}
}

