// Executed inside Photoshop by photoshop_bridge.py.
// __bridge_run__() evaluates the staged .jsx with $.writeln captured, writes the
// log + the value of a `__result__` variable to the output file, and returns it.

function __bridge_run__(inPath, outPath) {
    var log = [];
    var savedWrite = $.writeln;
    $.writeln = function (s) { log.push(String(s)); };

    var source = "";
    var error = null;
    var input = new File(inPath);
    input.encoding = "UTF-8";
    if (!input.open("r")) {
        error = "cannot read " + inPath;
    } else {
        source = input.read();
        input.close();
    }

    if (!error) {
        try {
            eval(source);
        } catch (e) {
            error = e.toString() + (e.line ? " (line " + e.line + ")" : "");
        }
    }
    $.writeln = savedWrite;

    var result = null;
    if (typeof __result__ !== "undefined") { result = __result__; }

    var text = log.join("\n");
    if (error) { text += (text ? "\n" : "") + "BRIDGE ERROR: " + error; }
    if (result !== null && typeof result !== "undefined") {
        text += (text ? "\n" : "") + "__result__ = " + result;
    }

    var output = new File(outPath);
    output.encoding = "UTF-8";
    if (output.open("w")) {
        output.write(text);
        output.close();
    }
    return text;
}
