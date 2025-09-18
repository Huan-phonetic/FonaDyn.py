// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, KTH Stockholm
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //

VRPContext {
	var <model; // The model - data, settings, buses, groups, etc
	var <controller; // The controller

	*new { | libname, server |
		^super.new.init(libname, server);
	}

	init { | libname, server |
		model = VRPModel(libname, server);
		controller = VRPController();
	}
}