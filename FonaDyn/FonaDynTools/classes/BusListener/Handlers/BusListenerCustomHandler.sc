// Copyright (C) 2016-2025 by Sten Ternström & Dennis J. Johansson, 
// Released under European Union Public License v1.2, at https://eupl.eu
// *** EUPL *** //
BusListenerCustomHandler {
	var mFn;

	*new { | fn |
		^super.new.init(fn);
	}

	init { | fn |
		mFn = fn;
	}

	dispatch { | data |
		mFn.(data);
	}

	free {
	}
}