// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //

VRPViewMaps {
	var <mView;
	var vrpMaps;
	var vrpViews;
	var mSideTitle = "FonaDyn Participant Map";
	var mLastIndex, mCurrentIndex;
	var mButtons;
	var modeButtons;
	var mTileStackButton;
	var mSaveImagesButton;
	var mLastPath;
	var mbStacked;
	var mLayerRequested;
	var iMainLayout;
	var mvLayout, mbHLayout;
	var prSettings;
	var bMapAdded = false;
	var stateTarget = 0; // 0-no target, 1=target requested, 2=target active, 3=target deactivate
	var mTargetDSM;
	classvar <bTileVertically = false;
	classvar <mAdapterUpdate;

	*new { | view |
		^super.new.init(view);
	}

	*configureTiledMaps { arg vertical;
		if (vertical.notNil, { bTileVertically = vertical } );
	}

	getMetricFromLayer { arg layerSymbol;
		var m=nil, vrpData;
		vrpData = vrpMaps.first.mVRPdata;
		if (vrpData.notNil, { m = vrpData.layers[layerSymbol].metric } );
		^m
	}

	init { | view |
		var btn, btn_state;

		modeButtons =
		[ [["now", Color.gray(0.7), Color.gray(0.4)],["NOW", Color.white, Color.gray(0.7)]],
		  [["twin", Color.gray(0.7), Color.green(0.5)], ["TWIN", Color.white, Color.green(0.75)]],
		  [["twin »", Color.gray(0.7), Color.green(0.5)], ["TWIN »", Color.white, Color.green(0.75)]],
		  [["before", Color.gray(0.7), Color.magenta(0.5)], ["BEFORE", Color.white, Color.magenta(0.75)]],
		  [["diff", Color.gray(0.7), Color.blue(0.65)], ["DIFF", Color.white, Color.blue(0.85)]],
		  [["smooth", Color.gray(0.7), Color.yellow(0.5)], ["SMOOTH", Color.white, Color.yellow(0.8)]]
		];

		mView = view;
		vrpMaps = [];
		vrpViews = [];
		mButtons = [];
		mbStacked = true;
		mvLayout = StackLayout.new([]);
		iMainLayout = 0;
		mLayerRequested = -1;
		prSettings = nil;
		mTargetDSM = nil;
		stateTarget = 0;
		mAdapterUpdate = { | menu, who, what, newValue |
			this.update(menu, who, what, newValue);
		};

		mTileStackButton = Button(mView, Rect());
		mTileStackButton
		.states_( [["□□□"], ["╒══╗"]] )  // Stack off, Stack on
		.value_(mbStacked.asInteger)
		.action_({ |b| this.setStackMode(b.value.asBoolean) });
		mSaveImagesButton = Button(mView, Rect());
		mSaveImagesButton
		.states_( [["Save images"]] )
		.action_( { | v |
			var rect = (vrpViews.first.bounds union: vrpViews.last.bounds).moveTo(0, mTileStackButton.bounds.height+7);
			vrpMaps.first.writeImage(mView, rect, vrpMaps.first.mLastPath, { | retPath | mLastPath = retPath });
		});
		this.addMap(VRPViewVRP.iNormal);  		// Normal map view for the first one
		this.setStackMode(true);				// Draw the signal on top of the grid
		~myVRPViewPlayer.setMapHandler(this);	// Mediate signal selection changes to the maps
		~getCurrentMetric = { arg sym; this.getMetricFromLayer(sym) };  // For Matlab, via VRPMetric.cMap
	}

	mapIndex { | mode |
		var a = [];
		vrpMaps do: ( { | m, i | a = a.add(m.mapMode) } );
		^a.indexOf(mode)
	}

	toggleLayout {
		bTileVertically = bTileVertically.not;
		this.layout;
	}

	toggleMap { | mode |
		var ix=nil, index;
		var newMode = mode;
		index = this.mapIndex(mode);

		// Let the two kinds of TWIN maps also toggle each other:
		case
		{ (mode == VRPViewVRP.iClone) and: { ( ix = this.mapIndex(VRPViewVRP.iCloneFloat)).notNil } }
			{ newMode = VRPViewVRP.iCloneFloat; index = ix }
		{ (mode == VRPViewVRP.iCloneFloat) and: { (ix = this.mapIndex(VRPViewVRP.iClone)).notNil } }
			{ newMode = VRPViewVRP.iClone; index = ix }
		; /* case */

		// Toggle the existence of a map with the given mode
		if (index.isNil,
			{
				this.addMap(newMode);
			},{
				this.removeMap(index);
			}
		);

		^index.isNil;
	}

	addMapViewVRP { arg mode, parentView, srcVRPmap;
		var win, resultMap;
		// A floating TWIN map needs its own parent window
		if (mode == VRPViewVRP.iCloneFloat, {
			win = Window.new(mSideTitle, Rect(500, 200, 800, 700));
			resultMap = VRPViewVRP(win.asView, srcVRPmap);

			// Copy also the grid mode and the cycle threshold to the new map
			if (srcVRPmap.mGridHorzGridSelectHz != resultMap.mGridHorzGridSelectHz, {
				resultMap.toggleHorzGrid();
			});
			resultMap.setCycleThreshold(srcVRPmap.getCycleThreshold);

			win.onClose_{ this.removeMap() };
			win.front;
		}, {  // Others don't
			resultMap = VRPViewVRP( parentView, srcVRPmap );
		});

		// If there is an active target overlay, trigger a \targetOverlay change
		if (stateTarget == 2, { stateTarget = 1 });
		^resultMap
	}

	addMap { | mode |
		var nextView, srcVRPmap, btn;

		btn = Button(mView);
		btn
		.states_(modeButtons[mode])
		.action_({|b| this.setActiveTab( b )})
		.maxWidth_(60);

		srcVRPmap = vrpMaps[0];

		if ([VRPViewVRP.iClone, VRPViewVRP.iCloneFloat, VRPViewVRP.iSmooth].indexOf(mode).notNil,
			{ // Find the iNormal map, for cloning
				vrpMaps do: { | m, i |
					if (m.mapMode == VRPViewVRP.iNormal,
						{ srcVRPmap = vrpMaps[i] })
				};
			}
		);

		nextView = CompositeView(mView, mView.bounds);

		// The map created first is always iNormal
		if (vrpViews.size == 0, { mode = VRPViewVRP.iNormal });

		if (mode != VRPViewVRP.iReference,
			{
				// Most maps are added last (rightmost)
				vrpViews = vrpViews.add (nextView);
				vrpMaps = vrpMaps.add ( this.addMapViewVRP(mode, nextView, srcVRPmap ) ) ;
				vrpMaps.last.mapMode_(mode);
				vrpMaps.last.setMapHandler(this);
				this.addDependant(vrpMaps.last);
				mLastIndex = vrpMaps.size - 1;
				mCurrentIndex = mLastIndex;
				mButtons = mButtons.add(btn);
			} , {
				// Reference map is always first (leftmost)
				vrpViews = vrpViews.insert (0, nextView);
				vrpMaps = vrpMaps.insert (0, VRPViewVRP( nextView, srcVRPmap )) ;
				vrpMaps.first.mapMode_(mode);
				vrpMaps.first.setMapHandler(this);
				this.addDependant(vrpMaps.first);
				mLastIndex = vrpMaps.size - 1;
				mCurrentIndex = 0;
				btn.mouseDownAction_( { arg b, x, y, mods, btnNum, clicks;
					// Double-click to toggle the existence of a target overlay map
					if (clicks == 2,
						{
							stateTarget = stateTarget + 1;
							if (stateTarget == 1, {
								mTargetDSM = vrpMaps.first.buildTargetDSM;
							});
						}
					);
				});
				mButtons = mButtons.insert(0, btn);
		});

		if (mode == VRPViewVRP.iDiff,
			{   var tempVRPdata, nC;
				// Make a new but empty VRPViewVRP-diff
				tempVRPdata = VRPDataVRP.new(bDiff: true);

				// Set its cluster counts to those of the BEFORE map
				nC = vrpMaps.first.mVRPdata.layers[\ClustersEGG].cCount;
				tempVRPdata.initClusteredLayers(iType: VRPSettings.iClustersEGG, nClusters: nC, bDiffMap: true);
				nC = vrpMaps.first.mVRPdata.layers[\ClustersPhon].cCount;
				tempVRPdata.initClusteredLayers(iType: VRPSettings.iClustersPhon, nClusters: nC, bDiffMap: true);

				vrpMaps.last.mVRPdata = tempVRPdata;
				vrpMaps.last.computeDiffs(vrpMaps[0].mVRPdata, vrpMaps[1].mVRPdata);  // We need safeguards for this
			}
		);

		if (mode == VRPViewVRP.iSmooth,
			{
				// Make a new but empty VRPViewVRP
				// The prSettings are needed to initialize the number of clusters
				vrpMaps.last.mVRPdata = VRPDataVRP.new(prSettings);
				vrpMaps.last.interpolateSmooth(srcVRPmap.mVRPdata);
			}
		);

		bMapAdded = true;
		btn.valueAction_(1);
		this.layout;
	} /* addMap */

	removeMap { | index=nil |
		index = index ?? { vrpViews.size - 1 };

		// Any floating TWIN map needs to have its parent window closed too
		if (index.isNil or: (vrpMaps[index].mapMode == VRPViewVRP.iCloneFloat), {
			Window.allWindows do: { | w | if (w.name == mSideTitle, { w.close } ) }
		});

		if ((vrpViews.size > 1)
			and: { vrpMaps[index].mapMode != VRPViewVRP.iNormal }, // Never remove the NOW map
			{
				this.removeDependant(vrpMaps[index]);
				this.changed(this, \mapWasDeleted, vrpMaps[index]);
				if (vrpMaps[index].mapMode == VRPViewVRP.iReference, { stateTarget = 0 } );
				vrpMaps.removeAt(index).close;
				vrpViews.removeAt(index).remove;
				mButtons.removeAt(index).remove;
				vrpMaps do: { |m, i|
					if (m.mapMode == VRPViewVRP.iNormal, {
						this.setActiveTab(mButtons[i])
					});
				};
		});
		this.layout;
	}

	layout {
		var theLayout;
		var vrpViewsFixed = [];

		// Avoid adding a floating TWIN map to the layout
		vrpMaps.do { | m, ix | if (m.mapMode != VRPViewVRP.iCloneFloat, {
			vrpViewsFixed = vrpViewsFixed.add(vrpViews[ix]);
			})
		};

		if (mbStacked,
			{
				mvLayout = StackLayout.new(*vrpViewsFixed);
				mvLayout.index = mCurrentIndex;
			}, {
				if (bTileVertically
					and: { [VRPViewMain.layoutGallery, VRPViewMain.layoutGalleryAll].indexOf(iMainLayout).isNil },
					{
						mvLayout = VLayout.new(*vrpViewsFixed);
						mTileStackButton.states_([["╞══╡"], ["╒══╗"]]);
					}, {
						mvLayout = HLayout.new(*vrpViewsFixed);
						mTileStackButton.states_([["□□□"], ["╒══╗"]]);
					}
				);
			vrpViewsFixed do: { |v| v.visible_(true) }; // StackLayout changes .visible
			}
		);

		if (vrpViews.size > 1, {
			mbHLayout = HLayout.new([mTileStackButton, stretch: 0, align: \left]);
			mButtons do: { |b, i| mbHLayout.add(b, stretch: 0, align: \left)};
			mbHLayout.add(nil, stretch: 10);
			mbHLayout.add(mSaveImagesButton, stretch: 0, align: \right);
			mbHLayout.margins_(5);
			theLayout = VLayout.new([mbHLayout, align: \left], mvLayout);
			mTileStackButton.visible_(true);
		}, {
			mTileStackButton.visible_(false);
			theLayout = mvLayout;
		});

		mSaveImagesButton.visible_((vrpViewsFixed.size > 1) and: mbStacked.not);

		theLayout
		.margins_(0)
		.spacing_(0);

		mView.layout = theLayout;
		mView.refresh;
	}

	setActiveTab { | btn |
		mCurrentIndex = mButtons.indexOf(btn);
		if (mbStacked, { mvLayout.index = mCurrentIndex });
	}

	setStackMode { | bStacked |
		mbStacked = bStacked;
		this.layout;
	}

	stash { | settings |
		mTileStackButton.value_(mbStacked);
		vrpMaps do: { | x |
			x.stash(settings);
		};
		mView.setProperty(\visible, settings.vrp.isVisible, true);
		// mView.visible_(settings.vrp.isVisible);
	}

	fetch { | settings |
		mbStacked = mTileStackButton.value.asBoolean;
		if (settings.waitingForStash, { this.stash(settings) });
		settings.vrp.isVisible = mView.visible;
		if (iMainLayout != settings.general.layout, 	// User changed the main layout
			{
				iMainLayout = settings.general.layout.abs;
				this.layout;
			}
		);
		vrpMaps do: { | x |
			x.fetch(settings);
		};
		prSettings = settings;
	}

	update { | menu, who, whatHappened, newValue |
		if (whatHappened == \selectLayer, {
			// Don't propagate this change, because it would cause recursion;
			// instead ask the other maps to switch layer.

			if ((
				( vrpMaps.indexOf(who) == mCurrentIndex )
				and:
				{ [VRPViewVRP.iClone, VRPViewVRP.iCloneFloat].indexOf(who.mapMode).isNil }
			),
			{
				mLayerRequested = newValue;
			});
			if (
				( vrpMaps[mCurrentIndex].mapMode == VRPViewVRP.iCloneFloat )
				and:
				{ who.mapMode == VRPViewVRP.iNormal },
				{
					vrpMaps[mCurrentIndex].setLayer(newValue);
				}
			);

		}, {
			// Propagate all other changes to the current maps
			this.changed(who, whatHappened, newValue);
		});
	}

	updateData { | data |
		var dsg = data.settings.general;
		var selected;

		// Emulate radio button behaviour
		mButtons do: ({ |b, i|
			b.value_((i==mCurrentIndex).asInteger);
			b.visible_(vrpViews.size > 1);
		});

		// Propagate layer selections as required
		if (mLayerRequested >= 0, {
			vrpMaps do: { | map, i |
				if ([VRPViewVRP.iClone, VRPViewVRP.iCloneFloat].indexOf(map.mapMode).isNil,
					{	map.setLayer(mLayerRequested) }
				);
			};
			mLayerRequested = -1;
		});

		// Let NOW know the selection of TWIN, if any
		// TWIN, if any, is always after NOW
		selected = nil;
		vrpMaps.reverse do: { | map, i |
			if ((map.mapMode == VRPViewVRP.iClone) or: {map.mapMode == VRPViewVRP.iCloneFloat},  { selected = map.layers } );
			if (selected.notNil, {
				if (map.mapMode == VRPViewVRP.iNormal, { map.setTwinLayers(selected) });
			});
		};

		if (dsg.guiChanged, {
			mView.background_(dsg.getThemeColor(\backPanel));
		});

		if (bMapAdded, {
			dsg.guiChanged = true; // Ask new maps to adopt the current color scheme
			bMapAdded = false;
		});

		case
		{ stateTarget == 1 }
			{
			data.player.targetDSM = mTargetDSM;
			mButtons.first.states = [["target", Color.gray(0.7), Color.magenta(0.5)], ["TARGET", Color.white, Color.magenta(0.75)]];
			stateTarget = 2;
			this.changed(this, \targetOverlay, mTargetDSM);
			}
		{ stateTarget == 3 }
			{
			data.player.targetDSM = mTargetDSM = nil;
			mButtons.first.states = [["before", Color.gray(0.7), Color.magenta(0.5)], ["BEFORE", Color.white, Color.magenta(0.75)]];
			stateTarget = 0;
			this.changed(this, \targetOverlay, nil);
			}
		;

		if (mbStacked or: bTileVertically, { data.vrp.mapsWidth = 1 },
			{
				data.vrp.mapsWidth = vrpMaps.size
			}
		);

		vrpMaps do: { | x, i |
			x.updateData(data);
		};
	} /* .updateData */

	close {
		this.removeMap();
		vrpViews do: { | x |
			x.close;
		};
		this.release;
	}
}