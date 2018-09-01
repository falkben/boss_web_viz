'''
These sets of classes hold the data that goes into the JSON for a neuroglancer link
'''

import json


class Neuroglancer:
    def __init__(self):
        self.layers = NeuroglancerLayers()
        self.nav = NeuroglancerNav()
        self.extra = NeuroglancerExtra()

    def get_json(self):
        data = {**self.layers.get(), **self.nav.get(), **self.extra.get()}
        return json.dumps(data)

    def add_layer(self, name, source, imagetype):
        self.layers.add_layer(name, source, imagetype)


class NeuroglancerLayers:
    def __init__(self):
        self.layers = {'layers': []}

    def get(self):
        return self.layers

    def add_layer(self, name, source, imagetype):
        newlayer = {name: {'source': source, 'type': imagetype}}
        self.layers['layers'].append(newlayer)


class NeuroglancerExtra:
    def __init__(self):
        self.layout = {'layout': 'xy'}

    def get(self):
        return self.layout

    def change_layout(self, new_layout):
        self.layout['layout'] = new_layout


class NeuroglancerNav:
    def __init__(self):
        self.nav = {}

    def get(self):
        return self.nav

    def add_pose(self, voxelSize, voxelCoordinates):
        self.nav['pose'] = {'position': {
            'voxelSize': voxelSize, 'voxelCoordinates': voxelCoordinates}}

    def add_zoom(self, zoom):
        self.nav['zoomFactor'] = zoom
