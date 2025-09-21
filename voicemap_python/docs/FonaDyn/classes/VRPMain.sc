// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //

/*
// FonaDyn (C) Sten Ternström, Dennis Johansson 2016-2025
// KTH Royal Institute of Technology, Stockholm, Sweden.
// For full details of using this software,
// please see the FonaDyn Handbook, and the class help files.
// The main entry point to the online help files is that of the class FonaDyn.
*/

VRPMain {
	classvar <mVersion = "3.3.0";
	// Graphics
	classvar mWindow = nil;
	var <mViewMain, mRect;

	// Data
	var <mContext;
	var mGUIRunning;
	var mMutexGUI;

	// Edit the character after $ to set the column delimiter in CSV files
	classvar <cListSeparator = $; ;

	// Clocks
	var mClockGuiUpdates; 	// Send updates to the GUI via this tempoclock
	var mClockControllers; 	// Clock used by the controllers to schedule fetches etc
	//	var mClockDebug;  		// Clock used to dump debug info at intervals

	// Sundry
	classvar <guiUpdateRate = 24;  // Must be an integer # frames/sec
	classvar <settingsArchiveSymbol;

	*new { arg bRerun = false, script=nil;
		^super.new.start(bRerun, script);
	}

	*screenScale {
		var point = 1.0@1.0;
		var rc= Window.availableBounds();
		point.x = rc.width / 1024.0;
		point.y = rc.height / 1024.0;
		^point
	}

	*openPanelPauseGUI{ arg okFunc, cancelFunc, multipleSelection = false, path;
		var fnDoneOK, fnDoneCancel;

		// Inhibit the GUI refresh cycle while a dialog box is open.
		// (Preempts warnings "scheduler queue is full")
		~dialogIsOpen = true;

		fnDoneOK = { | retNames |
			okFunc.(retNames);
			~dialogIsOpen = false;
		};

		fnDoneCancel = {
			cancelFunc.();
			~dialogIsOpen = false;
		};

		Dialog.openPanel(fnDoneOK, fnDoneCancel, multipleSelection, path);
	}

	*savePanelPauseGUI{ arg okFunc, cancelFunc, path, wantedSuffix="";
		var f, fnDoneOK, fnDoneCancel, fnSuffixIt;

		// Inhibit the GUI refresh cycle while a dialog box is open
		~dialogIsOpen = true;

		fnSuffixIt = { | name, wSuffix |
			var fullName;
			fullName = name;
			if (wSuffix.isEmpty.not and: (name.toLower.endsWith(".csv").not), {
				fullName = name ++ wSuffix;
			});
			fullName
		};

		fnDoneOK = { | retName |
			var aStr, fullName;
			// Add a filename suffix if requested
			fullName = fnSuffixIt.(retName, wantedSuffix);

			// Check that the file can be opened, without erasing it.
			if (File.exists(fullName), { aStr = "r+" }, { aStr = "w" });
			f = File.new(fullName, aStr);
			if (f.isOpen.not, {
				format("Could not save to %\n - is it open in another application?", fullName).warn;
				fnDoneCancel.();
			}, {
				f.close;
				okFunc.(fullName);
				~dialogIsOpen = false;
			})
		};

		fnDoneCancel = {
			cancelFunc.();
			~dialogIsOpen = false;
		};

		// The .csv file might be open e.g. in Excel, so check first
		Dialog.savePanel(fnDoneOK, fnDoneCancel, path);
	}

	postLicence {
		var texts = [
			"=========== FonaDyn Version % ============",
			"© 2017-2025 Sten Ternström, Dennis Johansson, KTH Royal Institute of Technology",
			"Licensed under European Union Public License v1.2, see ",
			~gLicenceLink ];

		format(texts[0], mVersion).postln;
		texts[1].postln;
		texts[2].postln;
		texts[3].postln;
	}

	createAlertSounds {
		SynthDef(\FonaDynBonk, {
			var env = Env([1, 1, 0], [1.0, 0.1]);
			var klank, klankenv;
			klank = Klank.ar(`[[247, 413, 1011, 1523, 2200], nil, [1, 0.7, 0.5, 0.4, 0.2]], Impulse.ar(0, 0, 0.05));
			klankenv = klank * EnvGen.ar(env, doneAction: Done.freeSelf);
			Out.ar(0, klankenv ! 2);
		}).add(mContext.model.libname);

		SynthDef(\FonaDynCrash, {
			var env = Env( [1, 1, 0.0, 0.0], [1.0, 0.01, 0.5] );
			var klank, strikes;
			strikes = Impulse.ar(20, 0, 0.05) * EnvGen.ar(env, doneAction: Done.freeSelf);
			klank = Klank.ar(`[[347, 713, 1211, 1523, 2703], nil, [1, 0.7, 0.5, 0.4, 0.2]], strikes);
			Out.ar(0, klank ! 2);
		}).add(mContext.model.libname);

		"FonaDyn: sonic alerts are activated.".postln;
	}

	update { arg theChanged, theChanger, whatHappened;
		if (whatHappened == \dialogSettings, {
			mContext.model.resetData;
		});
	}

	start { arg bRerun;
		var rcTmp;

		// Avoid creating multiple instances of FonaDyn
		if (~gVRPMain.notNil, {
			"A second instance can not be started.".warn;
			mWindow.front;
			^false
		});

		this.class.setMinorVersion();  // In FonaDynChangeLog.sc

		~gLicenceLink = "https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12";
		~gVRPMain = this;
		~dialogIsOpen = false;
		settingsArchiveSymbol = ("FonaDyn"++mVersion.replace($.,"")).asSymbol;

		// Set the important members
		mContext = VRPContext(\global, Server.default);
		mClockGuiUpdates = TempoClock(this.class.guiUpdateRate);	 // Maybe increase the queuesize here?
		mClockControllers = TempoClock(60, queueSize: 1024); 		 // Enough space for 512 entries
		//		mClockDebug = TempoClock(0.2);

		// Start the server
		mContext.model.server.boot;

		// Create the main window
		rcTmp = Window.availableBounds;
		mRect = rcTmp.setExtent(0.7*rcTmp.width, 0.85*rcTmp.height).moveTo(10, 0);
		mWindow = Window.new("FonaDyn", mRect, true, true);
		mWindow.setTopLeftBounds(mRect);
		mWindow.view.background_( Color.grey(0.85) );   // no effect?

		// Create the Main View
		mViewMain = VRPViewMain( mWindow.view );
		mViewMain.fetch(mContext.model.settings);
		mContext.model.resetData;

		mContext.model.server.doWhenBooted( {
			this.postLicence;
			this.createAlertSounds;
			// If we are supposed to use existing settings, get them and stash them
			if (bRerun, {
				if(mContext.model.settings.unarchive(settingsArchiveSymbol),  // from the archive
					{
						mViewMain.stash(mContext.model.settings);
						mViewMain.fetch(mContext.model.settings);
				})
			}, {
				// If the user has configured a start-up script, flag it for execution after boot
				mContext.model.settings.general.queueInitScript = VRPViewMainMenuInput.initScript.isString;
			});
		} );

		mWindow.onClose_ {
			var gd = mContext.model.data.general;
			var gs = mContext.model.settings.general;

			mContext.model.data.player.markForStop;	// Stop any ongoing playback
			if (gd.started, {
				gs.stop = true;
			});
			gs.start = false;
			Routine.new({this.updateData}).next; 	// Stop any ongoing analysis

			mGUIRunning = false;
			mClockControllers.stop;
			mClockGuiUpdates.stop;

			if (gs.saveSettingsOnExit, {
				gs.saveSettingsOnExit = false;
				mContext.model.data.settings.archive(settingsArchiveSymbol);
				"Settings archived; ".post;
			});
			~gVRPMain = nil;
			~gLicenceLink = nil;
			mViewMain.close;
			"FonaDyn was closed.".postln;
		};

		// Initiate GUI updates
		mMutexGUI = Semaphore();
		mGUIRunning = true;
		mClockGuiUpdates.sched(1, {
			var ret = if (mGUIRunning, 1, nil);
			Routine.new({this.updateData}).next;
			ret
		});

		// Force the main window to open at the recommended size
		// BUG? maXimize/Restore main window does not work after this,
		// except using Alt+Space, then X or R
		mWindow.view.fixedSize_(mRect.asSize);
		mWindow.front;

		// Then loosen the size
		mWindow.view.maxSize_(100000@100000);
		mWindow.view.minSize_(800@600);

/*		Exception.debug = true;
		mClockDebug.sched(1, {
		var ret = if (mGUIRunning, 1, nil);
		Routine.new({
		Main.gcInfo;
		// "Free: " + Main.totalFree.postln;
		}.defer ).next;
		ret
		});
*/
	} /* .start */

	guiUpdate {
		// Propagates the update to the views if the GUI is running
		var m = mContext.model;
		if ( mGUIRunning and: (~dialogIsOpen.not), {
			defer {
				if (mWindow.notNil and: { mWindow.isClosed.not }, {
					mViewMain.updateData(m.data);
					mViewMain.fetch(m.settings);
				});
				// Time when the refresh cycle is complete
				m.data.general.addTime(this.class.guiUpdateRate);
				mMutexGUI.signal;
			};
			// Time when the refresh cycle starts
			m.data.general.markTime;
			mMutexGUI.wait;

			// If a script has requested some new setting,
			// copy all settings into data.settings
			if (m.settings.waitingForStash, {
				m.settings.waitingForStash = false;
				m.resetData(true);
			});
		});
	}

	updateData {
		var cond = Condition();
		var c = mContext;
		var cs = c.controller;
		var m = c.model;
		var s = m.server;
		var d = m.data;
		var se = m.settings;
		var bm = m.busManager;
		var deltaTime;

		block { | break |
			if ( se.general.start, {
				se.general.start = false;
				Date.localtime.format("START %H:%M:%S, %Y-%m-%d").postln;

				// We should start the server!
				if (d.general.started or: d.general.starting, {
					d.general.error = "Unable to start the server as it is already started!";
					break.value; // Bail out
				});

				d.general.starting = true;
				this.guiUpdate(d); // Let the views know that we're starting the server

				if ( se.sanityCheck.not, {
					// Some check failed - bail out
					d.general.starting = false;
					d.general.started = false;
					break.value;
				});

				// Reset the data - grabbing the new settings
				m.resetData;
				d = m.data;

				// Wait for the server to fully boot
				s.bootSync(cond);

				// Allocate the groups
				value {
					var c = Condition();
					var sub_groups = { Group.basicNew(s) } ! 9;
					var main_group = Group.basicNew(s);
					var msgs = [main_group.newMsg(s), main_group.runMsg(false)]; // Ensure that the main group is paused immediately!
					msgs = msgs ++ ( sub_groups collect: { | g | g.newMsg(main_group, \addToTail) } ); // Create the rest normally

					// The order here is very important,
					// see the spreadsheet "VRP-code-diagrams.xlsx"
					m.groups.putPairs([
						\Main, main_group,
						\Input, sub_groups[0],
						\AnalyzeAudio, sub_groups[1],
						\CSDFT, sub_groups[2],
						\PostProcessing, sub_groups[3],
						\SampEn, sub_groups[4],
						\Cluster, sub_groups[5],
						\ClusterPhon, sub_groups[6],
						\Scope, sub_groups[7],
						\Output, sub_groups[8],
					]);

					// Send the bundle and sync
					s.sync(c, msgs);
				};

				// Create the controllers
				cs.io = VRPControllerIO(m.groups[\Input], m.groups[\Output], m.groups[\Main], d);
				cs.cluster = VRPControllerCluster(m.groups[\Cluster], d);
				cs.clusterPhon = VRPControllerClusterPhon(m.groups[\ClusterPhon], d);
				cs.csdft = VRPControllerCSDFT(m.groups[\CSDFT], d);
				cs.sampen = VRPControllerSampEn(m.groups[\SampEn], d);
				cs.scope = VRPControllerScope(m.groups[\Scope], d);
				cs.vrp = VRPControllerVRP(m.groups[\AnalyzeAudio], d);
				cs.postp = VRPControllerPostProcessing(m.groups[\PostProcessing], d);

				// Find out what buses are required and allocate them
				cs.asArray do: { | c | c.requires(bm); };
				bm.allocate();
				s.sync;
				if (m.settings.general.enabledDiagnostics, {
				    bm.debug;  // Post the bus numbers for inspection
				});

				// Prepare all controllers and sync
				cs.asArray do: { | x | x.prepare(m.libname, s, bm, mClockControllers); };
				s.sync;

				// Start all controllers and sync
				cs.asArray do: { | x | x.start(s, bm, mClockControllers); };
				s.sync;

				// Resume the main group so all synths can start!
				m.groups[\Main].run;
				d.general.started = true;
				d.general.starting = false;
				d.general.pause = 0;
			}); // End start

			if (d.general.pause == 1, {
				"Pausing - ".post;
				m.groups[\Main].run(false);
				s.sync;
				d.general.pause = 2;
			});

			if (d.general.pause == 3, {
				m.groups[\Main].run(true);
				d.general.pause = 0;
				Date.localtime.format("resumed %H:%M:%S").postln;
			});

			// Either user wants to stop, or we reached EOF
			// - make sure we're not already stopping or have stopped the server.
			if (se.general.stop or: (d.io.eof and: d.general.stopping.not and: d.general.started), {
				se.general.stop = false;
				Date.localtime.format("STOP  %H:%M:%S").postln;

				// Perform sanity checks, if they fail -> bail out
				if (d.general.started.not, {
					d.general.error = "Unable to stop: the server has not yet been started.";
					break.value; // Bail out
				});

				d.general.stopping = true;

				// Pause the main group
				m.groups[\Main].run(false);

				// Stop the controllers and sync
				cs.asArray do: { | x | x.stop; };

				this.guiUpdate(d); // Let the views know that we're stopping the server

				cs.asArray do: { | x | x.sync; };

				// Free the buses & groups
				m.busManager.free;
				m.groups[\Main].free;
				m.groups.clear;

				// Done
				s.sync;
				d.general.started = false;
				d.general.stopping = false;
			}); // End stop
		}; // End block

		this.guiUpdate(d);

	}

}

