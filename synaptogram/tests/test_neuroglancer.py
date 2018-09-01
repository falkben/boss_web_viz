import pytest

from ..neuroglancer import Neuroglancer, NeuroglancerLayers, NeuroglancerExtra, NeuroglancerNav


class TestNeuroglancer(object):

    def test_init_neuroglancer(self):
        ng = Neuroglancer()
        assert ng.nav.nav == {}
        assert ng.layers.layers == {'layers': []}
        assert ng.extra.layout == {'layout': 'xy'}

    def test_json_empty_neuroglancer_obj(self):
        ng = Neuroglancer()
        json_data = ng.get_json()
        expect_data = '{"layers": [], "layout": "xy"}'
        assert json_data == expect_data


class TestNeuroglancerLayers(object):

    def test_init_neuroglancer_layers(self):
        ngl = NeuroglancerLayers()
        assert ngl.layers == {'layers': []}

    def test_neuroglancer_layer(self):
        ngl = NeuroglancerLayers()
        name = 'synapsinR_7thA'
        source = 'boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000'
        image_type = 'image'
        ngl.add_layer(name, source, image_type)
        assert ngl.layers['layers'][0][name] == {
            'source': source, 'type': image_type}
