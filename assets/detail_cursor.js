/**
 * Synced hover cursor for the activity detail tab.
 * Hovering any detail chart (or the map) draws a vertical line at that km on every
 * chart and places a dot on the map at the matching GPS position.
 */
(function () {
    var CHART_IDS = ["detail-hr-pace", "detail-hr", "detail-pace", "detail-altitude", "detail-cum-elevation", "detail-power-cad"];
    var CURSOR = "km-cursor";
    var lastKm;

    function plotDiv(id) {
        var el = document.getElementById(id);
        return el ? el.querySelector(".js-plotly-plot") : null;
    }

    // Index of the track point whose distance is closest to km (binary search)
    function nearestIndex(dist, km) {
        var lo = 0, hi = dist.length - 1;
        while (lo < hi) {
            var mid = (lo + hi) >> 1;
            if (dist[mid] < km) lo = mid + 1; else hi = mid;
        }
        if (lo > 0 && Math.abs(dist[lo - 1] - km) < Math.abs(dist[lo] - km)) lo -= 1;
        return lo;
    }

    function updateCharts(km) {
        CHART_IDS.forEach(function (id) {
            var gd = plotDiv(id);
            if (!gd || !gd.layout || !window.Plotly) return;
            var shapes = (gd.layout.shapes || []).filter(function (s) { return s.name !== CURSOR; });
            if (km !== null) {
                shapes.push({
                    type: "line", name: CURSOR, xref: "x", yref: "paper",
                    x0: km, x1: km, y0: 0, y1: 1, layer: "above",
                    line: { color: "rgba(255,255,255,0.8)", width: 1.5 },
                });
            }
            window.Plotly.relayout(gd, { shapes: shapes });
        });
    }

    function updateMap(km, track) {
        var gd = plotDiv("detail-map");
        if (!gd || !gd.data || !window.Plotly) return;
        var idx = [];
        gd.data.forEach(function (t, i) { if (t.name === CURSOR) idx.push(i); });
        if (!idx.length) return;
        var lat = [], lon = [];
        if (km !== null && track && track.dist && track.dist.length) {
            var i = nearestIndex(track.dist, km);
            lat = [track.lat[i]];
            lon = [track.lon[i]];
        }
        window.Plotly.restyle(gd, {
            lat: idx.map(function () { return lat; }),
            lon: idx.map(function () { return lon; }),
        }, idx);
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        detail: {
            syncCursor: function (hrPace, hr, pace, alt, cumElev, powCad, mapHover, track) {
                var trig = (window.dash_clientside.callback_context.triggered || [])[0];
                if (!trig) return window.dash_clientside.no_update;

                var km = null;
                var pts = trig.value && trig.value.points;
                if (pts && pts.length) {
                    if (trig.prop_id.indexOf("detail-map.") === 0) {
                        // Map points carry [pace label, distance km] in customdata
                        var cd = pts[0].customdata;
                        km = Array.isArray(cd) && typeof cd[1] === "number" ? cd[1] : null;
                    } else if (typeof pts[0].x === "number") {
                        km = pts[0].x;
                    }
                }
                if (km === lastKm) return window.dash_clientside.no_update;
                lastKm = km;

                updateCharts(km);
                updateMap(km, track);
                return window.dash_clientside.no_update;
            },
        },
    });
})();
