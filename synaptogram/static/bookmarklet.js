javascript: (function () {
    var data_panel = document.getElementsByClassName('rendered-data-panel');
    if (data_panel.length == 0) {
        data_panel = document.getElementsByClassName('neuroglancer-rendered-data-panel neuroglancer-panel neuroglancer-noselect');
    }
    if (data_panel.length == 0) {
        window.open('http://localhost:8080/sgram_from_ndviz?url=' + encodeURIComponent(location.href))
    }
    var paneldiv = data_panel[0];

    var clientHeight = paneldiv.clientHeight;
    var clientWidth = paneldiv.clientWidth;

    var zoomfactor = window.viewer.navigationState.zoomFactor.value;
    var voxelSize = window.viewer.navigationState.pose.position.voxelSize.size;

    var spatialcoords = window.viewer.navigationState.pose.position.spatialCoordinates;

    var coords = [];
    for (i = 0; i < spatialcoords.length; i++) {
        coords.push(spatialcoords[i] / voxelSize[i]);
    }

    var datawidth = clientWidth * zoomfactor / voxelSize[0];
    var dataheight = clientHeight * zoomfactor / voxelSize[1];

    var xextents = [Math.round(coords[0] - datawidth / 2), Math.round(coords[0] + datawidth / 2)];
    var yextents = [Math.round(coords[1] - dataheight / 2), Math.round(coords[1] + dataheight / 2)];

    window.open('http://ndwebtools.neurodata.io/sgram_from_ndviz?xextent=' + xextents + '&yextent=' + yextents + '&coords=' + coords + '&url=' + encodeURIComponent(location.href))
}
)();
