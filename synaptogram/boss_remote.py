'''Class for handling requests to the BOSS'''

import requests


BOSS_URL = 'https://api.boss.neurodata.io'
BOSS_VERSION = "v1"

# documentation: https://docs.theboss.io/docs


class BossRemote:

    def __init__(self, request, auth_type='Bearer'):
        self.boss_url = BOSS_URL
        if self.boss_url[-1] != '/':
            self.boss_url += '/'
        self.boss_url += '{}/'.format(BOSS_VERSION)

        token = request.session.get('access_token')  # Bearer token
        self.token = token
        self.session = requests.Session()

        self.auth_type = auth_type

    def get(self, url, input_headers={}):
        # all get requests should come through here

        # strip the leading slash if there is one
        if url[0] == '/':
            url = url[1:]
        if len(input_headers) > 0:
            headers = input_headers
        else:
            headers = {}
        headers['Authorization'] = '{} {}'.format(self.auth_type, self.token)
        r = self.session.get("{}{}".format(
            self.boss_url, url), headers=headers)
        r.raise_for_status()
        return r

    def post(self, url, data=None, input_headers={}):
        # strip the leading slash if there is one
        if url[0] == '/':
            url = url[1:]
        if len(input_headers) > 0:
            headers = input_headers
        else:
            headers = {}
        headers['Authorization'] = '{} {}'.format(self.auth_type, self.token)
        r = self.session.post("{}{}".format(
            self.boss_url, url), data=data, headers=headers)
        r.raise_for_status()

    def delete(self, url, input_headers={}):
        # strip the leading slash if there is one
        if url[0] == '/':
            url = url[1:]
        if len(input_headers) > 0:
            headers = input_headers
        else:
            headers = {}
        headers['Authorization'] = '{} {}'.format(self.auth_type, self.token)
        r = self.session.delete("{}{}".format(
            self.boss_url, url), headers=headers)
        r.raise_for_status()

    def get_collections(self):
        get_url = 'collection/'
        r = self.get(get_url, {'Accept': 'application/json'})
        resp = r.json()

        if resp['collections'] == []:
            return None

        return sorted(resp['collections'], key=str.lower)

    def get_experiments(self, coll):
        get_url = 'collection/{}/'.format(coll)
        r = self.get(get_url, {'Accept': 'application/json'})
        resp = r.json()

        if resp['experiments'] == []:
            return None

        return sorted(resp['experiments'], key=str.lower)

    def get_channels(self, coll, exp):
        # returns all the channels in an experiment
        get_url = 'collection/{}/experiment/{}/channels/'.format(coll, exp)
        r = self.get(get_url, {'Accept': 'application/json'})
        resp = r.json()

        if resp['channels'] == []:
            return None

        return sorted(resp['channels'], key=str.lower)

    def get_exp_info(self, coll, exp):
        # returns info about the experiment
        # https://api.theboss.io/v1/collection/:collection/experiment/:experiment/
        exp_url = "collection/{}/experiment/{}/".format(
            coll, exp)
        r = self.get(exp_url, {'Accept': 'application/json'})
        return r.json()

    def get_ch_info(self, coll, exp, ch):
        # returns info about the channel
        get_url = 'collection/{}/experiment/{}/channel/{}'.format(
            coll, exp, ch)
        r = self.get(get_url, {'Accept': 'application/json'})
        return r.json()

    def get_coordinate_frame(self, coll, exp, exp_info=None):
        # get a coord frame for an experiment

        # https://api.theboss.io/v1/coord/:coordinate_frame
        # this gets us the coord frame name:
        if exp_info is None:
            exp_info = self.get_exp_info(coll, exp)

        coord_frame_name = exp_info['coord_frame']

        get_url = 'coord/{}'.format(coord_frame_name)
        r = self.get(get_url, {'Accept': 'application/json'})

        # should contain these data:
        # "x_start": 0,
        # "x_stop": 1000,
        # "y_start": 0,
        # "y_stop": 1000,
        # "z_start": 0,
        # "z_stop": 500,
        # "x_voxel_size": 1.0,
        # "y_voxel_size": 1.0,
        # "z_voxel_size": 1.0,
        # "voxel_unit": "nanometers",
        return r.json()

    def get_exp_metadata(self, coll, exp):
        # get all exp metadata keys
        get_url = 'meta/{}/{}'.format(coll, exp)
        r = self.get(get_url, {'Accept': 'application/json'})
        resp = r.json()

        # iterate through keys and get them all individually
        keys = resp['keys']
        metadata = {}
        for key in keys:
            metadata[key] = self.get_exp_metadata_key(coll, exp, key)['value']

        return metadata

    def get_exp_metadata_key(self, coll, exp, key):
        # get metadata value for key
        get_url = 'meta/{}/{}/?key={}'.format(coll, exp, key)
        r = self.get(get_url, {'Accept': 'application/json'})
        return r.json()

    def get_permissions(self, coll, exp=None, ch=None):
        params = '&'.join(
            [p + '=' + k for p, k in zip(['collection', 'experiment', 'channel'], [coll, exp, ch]) if k is not None])
        get_url = 'permissions/?{}'.format(params)
        r = self.get(get_url, {'Accept': 'application/json'})
        return r.json()

    def start_downsample(self, coll, exp, ch):
        post_url = '/'.join(('downsample', coll, exp, ch))
        self.post(post_url)

    def stop_downsample(self, coll, exp, ch):
        del_url = '/'.join(('downsample', coll, exp, ch))
        self.delete(del_url)

    def get_downsample_status(self, coll, exp, ch):
        get_url = '/'.join(('downsample', coll, exp, ch))
        r = self.get(get_url, {'Accept': 'application/json'})
        return r.json()
