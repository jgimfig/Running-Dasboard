/**
 * Converts a day-offset value from the date slider into a readable date string.
 * The base date is read from the hidden #slider-meta element (data-min-date attribute).
 */
window.dayOffsetToDate = function (value) {
    var meta = document.getElementById("slider-meta");
    if (!meta) return value;
    var raw = meta.getAttribute("data-min-date");
    if (!raw) return value;
    // Parse as UTC midnight to avoid timezone shifts
    var parts = raw.split("-");
    var base = Date.UTC(parseInt(parts[0]), parseInt(parts[1]) - 1, parseInt(parts[2]));
    var ts = base + Math.round(value) * 86400000;
    var d = new Date(ts);
    var dd = String(d.getUTCDate()).padStart(2, "0");
    var mm = String(d.getUTCMonth() + 1).padStart(2, "0");
    var yy = String(d.getUTCFullYear()).slice(-2);
    return dd + "/" + mm + "/" + yy;
};
