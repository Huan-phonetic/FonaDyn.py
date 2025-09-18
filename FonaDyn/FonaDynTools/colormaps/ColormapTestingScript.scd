// Color map testing script

(
w = Window.new(bounds: Rect(10, 200, 450, 80));
w.front;
w.background_(Color.white);
r = Rect(10, 10, 360, 50);
u = UserView(w, r);
d = thisProcess.platform.userExtensionDir +/+ "FonaDynTools/colormaps";
f = "FonaDynDefault256.csv";
w.name_(f);
t = Font.new("Arial", bold: true, size: 12);

u.drawFunc_{
	var c = 5;
	m = ColorMap.new();
	m.load(d +/+ f, true);
	p = m.smoothPaletteFunc;
	// p = m.steppedPaletteFunc(c);
	Pen.use {
		var nColors = r.width.half;
		var pos, rc, offs = 0;
		nColors.do ({ |k|
			rc = Rect(r.left+(k*2), r.top, 3, r.height) & r;
			pos = k / nColors.asFloat;
			// pos = (pos*(c)).floor/(c);
			Pen.color = p.(pos);
			Pen.addRect(rc);
			Pen.fill;
		});
		Pen.stringLeftJustIn(" " + f ++ ", smooth palette", r, t, Color.white);
	};
};

u.refresh;
)