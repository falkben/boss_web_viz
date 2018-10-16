javascript: (function () {
    let ndwt = 'https://ndwt.neurodata.io/';

    let data_panel = document.getElementsByClassName('rendered-data-panel');
    if (data_panel.length == 0) {
        data_panel = document.getElementsByClassName('neuroglancer-rendered-data-panel neuroglancer-panel neuroglancer-noselect');
    }

    let state = window.viewer.state.toJSON();

    if (typeof state.layers == 'undefined') {
        window.open(ndwt);
    }

    let idx = 0;
    let source = '';
    if (typeof (state.selectedLayer) != 'undefined') {
        idx = state.selectedLayer;
        source = state.layers[idx.layer].source
    } else {
        let L = state.layers;
        source = Object.entries(L)[0][1].source
    }

    let paneldiv = data_panel[0];

    let clientHeight = paneldiv.clientHeight;
    let clientWidth = paneldiv.clientWidth;

    let zoomfactor = window.viewer.navigationState.zoomFactor.value;
    let voxelSize = window.viewer.navigationState.pose.position.voxelSize.size;

    let spatialcoords = window.viewer.navigationState.pose.position.spatialCoordinates;

    let coords = [];
    for (i = 0; i < spatialcoords.length; i++) {
        coords.push(spatialcoords[i] / voxelSize[i]);
    }

    let datawidth = clientWidth * zoomfactor / voxelSize[0];
    let dataheight = clientHeight * zoomfactor / voxelSize[1];

    let xextents = [Math.round(coords[0] - datawidth / 2), Math.round(coords[0] + datawidth / 2)];
    let yextents = [Math.round(coords[1] - dataheight / 2), Math.round(coords[1] + dataheight / 2)];


    window.open(ndwt + 'sgram_from_ndviz?xextent=' + xextents + '&yextent=' + yextents + '&coords=' + coords + '&source=' + source)
})();