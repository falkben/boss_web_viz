import time
import re
import zipfile
import os
from io import BytesIO

import requests

from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse, FileResponse
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.contrib import messages

import blosc
import tifffile as tiff

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from .forms import CutoutForm


# All the actual views:

def index(request):
    if request.user.is_authenticated():
        username = get_username(request)
    else:
        username = ''
    request.session['next'] = 'synaptogram:coll_list'
    return render(request, 'synaptogram/index.html', {'username': username})


@login_required
def coll_list(request):
    add_url = 'collection/'
    url, headers = get_boss_request(request, add_url)
    resp = get_resp_from_boss(request, url, headers)
    request.session['next'] = 'synaptogram:coll_list'
    if resp == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')
    collections = sorted(resp['collections'], key=str.lower)
    username = get_username(request)
    context = {'collections': collections, 'username': username}
    return render(request, 'synaptogram/coll_list.html', context)


@login_required
def exp_list(request, coll):
    add_url = 'collection/{}/experiment/'.format(coll)
    url, headers = get_boss_request(request, add_url)
    resp = get_resp_from_boss(request, url, headers)
    request.session['next'] = '/exp_list/' + coll
    if resp == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')
    experiments = sorted(resp['experiments'], key=str.lower)

    username = get_username(request)
    context = {'coll': coll, 'experiments': experiments, 'username': username}
    return render(request, 'synaptogram/exp_list.html', context)


@login_required
def cutout(request, coll, exp):
    # we need the channels to fill the form
    channels = get_all_channels(request, coll, exp)
    if channels is None:
        return redirect(reverse('synaptogram:exp_list', args={coll}))
    elif channels == 'authentication failure':
        request.session['next'] = '/'.join('/cutout', coll, exp)
        return redirect('/openid/openid/KeyCloak')

    # getting the coordinate frame limits for the experiment:
    coord_frame = get_coordinate_frame(request, coll, exp)
    if coord_frame == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')
    # important stuff out of coord_frame:
        # "x_start": 0,
        # "x_stop": 1000,
        # "y_start": 0,
        # "y_stop": 1000,
        # "z_start": 0,
        # "z_stop": 500

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:

        form = CutoutForm(request.POST, channels=channels, limits=coord_frame)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            x = str(form.cleaned_data['x_min']) + \
                ':' + str(form.cleaned_data['x_max'])
            y = str(form.cleaned_data['y_min']) + \
                ':' + str(form.cleaned_data['y_max'])
            z = str(form.cleaned_data['z_min']) + \
                ':' + str(form.cleaned_data['z_max'])

            channels = form.cleaned_data['channels']

            pass_params_d = {'coll': coll, 'exp': exp, 'x': x,
                             'y': y, 'z': z, 'channels': ','.join(channels)}
            pass_params = '&'.join(['%s=%s' % (key, value)
                                    for (key, value) in pass_params_d.items()])
            params = '?' + pass_params

            # redirect to a new URL:
            end_path = form.cleaned_data['endpoint']
            if end_path == 'sgram':
                return HttpResponseRedirect(
                    reverse('synaptogram:sgram') + params)
            elif end_path == 'cut_urls':
                return HttpResponseRedirect(
                    reverse('synaptogram:cut_url_list') + params)
            elif end_path == 'ndviz':
                return HttpResponseRedirect(
                    reverse('synaptogram:ndviz_url_list') + params)
            elif end_path == 'tiff_stack':
                return HttpResponseRedirect(
                    reverse('synaptogram:tiff_stack') + params)

    # if a GET (or any other method) we'll create a blank form
    else:
        q = request.GET
        x_param = q.get('x')
        if x_param is not None:
            x, y, z = xyz_from_params(q)
            x_rng, y_rng, z_rng = create_voxel_rng(x, y, z)
            form = CutoutForm(channels=channels,
                              initial={'x_min': str(x_rng[0]), 'y_min': str(y_rng[0]), 'z_min': str(z_rng[0]),
                                       'x_max': str(x_rng[1]), 'y_max': str(y_rng[1]), 'z_max': str(z_rng[1])},
                              limits=coord_frame)
        else:
            form = CutoutForm(channels=channels,
                              limits=coord_frame)
    username = get_username(request)
    context = {'form': form, 'coll': coll, 'exp': exp, 'channels': channels,
               'username': username, 'coord_frame': sorted(coord_frame.items())}
    return render(request, 'synaptogram/cutout.html', context)


@login_required
def cut_url_list(request):
    q = request.GET
    coll, exp, x, y, z, channels = process_params(q)

    base_url, headers = get_boss_request(request)
    urls = ret_cut_urls(request, base_url, coll, exp, x, y, z, channels)

    channel_cut_list = zip(channels, urls)

    context = {'channel_cut_list': channel_cut_list, 'coll': coll, 'exp': exp}
    return render(request, 'synaptogram/cut_url_list.html', context)

# this generates an ndviz url for the channel/experiment/collection as inputs


@login_required
def ndviz_url(request, coll, exp, channel):
    base_url, headers = get_boss_request(request)
    if channel is None:
        channels = get_all_channels(request, coll, exp)
    else:
        channels = [channel]
    url, _ = ret_ndviz_urls(request, base_url, coll, exp, channels)
    return redirect(url[0])


@login_required
def ndviz_url_list(request):
    q = request.GET
    coll, exp, x, y, z, channels = process_params(q)
    base_url, headers = get_boss_request(request, '')
    urls, z_vals = ret_ndviz_urls(
        request, base_url, coll, exp, channels, x, y, z)

    channel_ndviz_list = zip(z_vals, urls)
    context = {
        'channel_ndviz_list': channel_ndviz_list,
        'coll': coll,
        'exp': exp}
    return render(request, 'synaptogram/ndviz_url_list.html', context)


@login_required
def tiff_stack(request):
    q = request.GET
    coll, exp, x, y, z, channels = process_params(q)

    urls = []
    for ch in channels:
        # create links to go to a method that will download the TIFF images
        # inside each channel
        urls.append(
            reverse(
                'synaptogram:tiff_stack_channel',
                args=(
                    coll,
                    exp,
                    x,
                    y,
                    z,
                    ch)))

    # or package the images and create links for the images
    channels_arg = ','.join(channels)

    return render(request, 'synaptogram/tiff_url_list.html', {'urls': urls,
                                                              'coll': coll, 'exp': exp, 'x': x, 'y': y, 'z': z, 'channels': channels_arg})


@login_required
def tiff_stack_channel(request, coll, exp, x, y, z, channel):
    img_data, cut_url = get_chan_img_data(request, coll, exp, channel, x, y, z)
    if img_data == 'authentication failure' or img_data == 'incorrect cutout arguments':
        return redirect(reverse('synaptogram:cutout', args=(coll, exp)))
    obj = BytesIO()
    tiff.imsave(obj, img_data, metadata={'cut_url': cut_url})

    fname = '_'.join(
        (coll, exp, str(x), str(y), str(z), channel)).replace(
        ':', '_') + '.tiff'
    response = HttpResponse(obj.getvalue(), content_type='image/TIFF')
    response['Content-Disposition'] = 'attachment; filename="' + fname + '"'
    return response


@login_required
def zip_tiff_stacks(request, coll, exp, x, y, z, channels):
    fname = 'media/' + \
        '_'.join(
            (coll, exp, str(x), str(y), str(z))).replace(
            ':', '_') + '.zip'

    channels = channels.split(',')

    try:
        os.remove(fname)
    except OSError:
        pass

    with zipfile.ZipFile(fname, mode='x', allowZip64=True) as myzip:
        for ch in channels:
            img_data, cut_url = get_chan_img_data(
                request, coll, exp, ch, x, y, z)
            if img_data == 'authentication failure' or img_data == 'incorrect cutout arguments':
                raise Exception
            fn = 'media/' + \
                '_'.join(
                    (coll, exp, str(x), str(y), str(z), ch)).replace(
                    ':', '_') + '.tiff'
            tiff.imsave(fn, img_data, metadata={'cut_url': cut_url})

            # running out of memory so I am not doing this anymore:
            # image = tiff.imread(fname)
            # np.testing.assert_array_equal(image, img_data)

            myzip.write(fn, arcname=fn.strip('media/'))

    response = FileResponse(open(fname, 'rb'))
    response['Content-Disposition'] = 'attachment; filename="' + \
        fname.strip('media/') + '"'
    return response


@login_required
def sgram(request):
    q = request.GET
    coll, exp, x, y, z, channels = process_params(q)
    return plot_sgram(request, coll, exp, x, y, z, channels)


@login_required
def sgram_from_ndviz(request):
    url = request.GET.get('url')
    coll, exp = parse_ndviz_url(request, url)

    if coll == 'incorrect source':
        return redirect('synaptogram:coll_list')
    elif coll == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')

    x = ':'.join(request.GET.get('xextent').split(','))
    y = ':'.join(request.GET.get('yextent').split(','))
    coords = request.GET.get('coords').split(',')
    z = ':'.join((coords[-1], str(int(coords[-1]) + 1)))

    # go to form to let user decide what they want to do
    pass_params_d = {'x': x, 'y': y, 'z': z}
    pass_params = '&'.join(['%s=%s' % (key, value)
                            for (key, value) in pass_params_d.items()])
    params = '?' + pass_params
    return HttpResponseRedirect(
        reverse('synaptogram:cutout', args=(coll, exp)) + params)
    #redirect('synaptogram:cutout', coll=coll,exp=exp)


@login_required
def channel_detail(request, coll, exp, channel):
    add_url = '/'.join(('collection', coll, 'experiment',
                        exp, 'channel', channel))
    url, headers = get_boss_request(request, add_url)
    resp = get_resp_from_boss(request, url, headers)
    request.session['next'] = '/exp_list/' + coll
    if resp == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')
    ch_props = resp

    add_url = 'permissions/?' + '&'.join(
        ('collection={}'.format(coll),
         'experiment={}'.format(exp),
         'channel={}'.format(channel)))
    url, headers = get_boss_request(request, add_url)
    resp = get_resp_from_boss(request, url, headers)
    request.session['next'] = '/exp_list/' + coll
    if resp == 'authentication failure':
        return redirect('/openid/openid/KeyCloak')
    permissions = resp
    perm_sets = permissions['permission-sets']

    coord_frame = get_coordinate_frame(request, coll, exp)
    base_url = get_boss_request(request)
    ndviz_url, _ = ret_ndviz_urls(request, base_url, coll, exp, [channel])

    return render(request, 'synaptogram/channel_detail.html',
                  {'coll': coll, 'exp': exp, 'channel': channel, 'channel_props': ch_props, 'permissions': perm_sets,
                   'ndviz_url': ndviz_url[0]})


@login_required
def start_downsample(request, coll, exp, channel):
    add_url = '/'.join(('downsample', coll, exp, channel))
    url, headers = get_boss_request(request, add_url)
    S = get_boss_session(request)
    r = S.post(url, headers=headers)
    if r.status_code != 201:
        print("Error: Request failed!")
        print("Received {} from server.".format(r.status_code))
    else:
        print("Downsample request sent for {}".format(url))
    return HttpResponseRedirect(
        reverse('synaptogram:channel_detail', args=(coll, exp, channel)))


@login_required
def stop_downsample(request, coll, exp, channel):
    add_url = '/'.join(('downsample', coll, exp, channel))
    url, headers = get_boss_request(request, add_url)
    S = get_boss_session(request)
    r = S.delete(url, headers=headers)
    if r.status_code != 204:  # 409 is the error if res. not found
        print("Error: Request failed!")
        print("Received {} from server.".format(r.status_code))
        # add messages.error
    else:
        print("Downsample delete request sent for {}".format(url))
    return HttpResponseRedirect(
        reverse('synaptogram:channel_detail', args=(coll, exp, channel)))


def set_sess_exp(request):
    id_token = request.session.get('id_token')
    epoch_time_KC = id_token['exp']
    epoch_time_loc = round(time.time())  # + time.timezone
    new_exp_time = epoch_time_KC - epoch_time_loc
    if new_exp_time < 0:  # this really shouldn't happen because we expire sessions now
        return 'expire time in past'
    else:
        request.session.set_expiry(new_exp_time)
        return 0


def get_username(request):
    return request.session.get('userinfo')['name']


# All the interaction with the Boss:

def get_boss_request(request, add_url=''):
    api_key = request.session.get('access_token')
    # api key should always be present because we should be calling
    # get_boss_request from views with @login_required
    boss_url = 'https://api.boss.neurodata.io/v1/'
    headers = {'Authorization': 'Bearer {}'.format(api_key)}
    url = boss_url + add_url
    return url, headers


def get_boss_session(request):
    if 'boss_sess' not in request.session:
        request.session['boss_sess'] = requests.Session()
        request.session['boss_sess'].stream = False
    return request.session['boss_sess']


def get_resp_from_boss(request, url, headers):
    S = get_boss_session(request)
    r = S.get(url, headers=headers)
    if r.status_code == 500:
        messages.error(request, 'server error')
        return 'server error'
    try:
        resp = r.json()
        if ('detail' in resp and resp['detail'] == 'Invalid Authorization header. Unable to verify bearer token') \
                or set_sess_exp(request) == 'expire time in past':
            return 'authentication failure'  # no message - we just redirecto to keycloak
        elif 'message' in resp and 'Incorrect cutout arguments' in resp['message']:
            messages.error(request, resp['message'])
            return 'incorrect cutout arguments'
        return resp
    except ValueError:
        # must be blosc data - return it
        return r


def get_all_channels(request, coll, exp):
    add_url = 'collection/{}/experiment/{}/channels/'.format(coll, exp)
    url, headers = get_boss_request(request, add_url)
    resp = get_resp_from_boss(request, url, headers)
    if resp == 'authentication failure':
        return resp  # need to handle this better
    elif resp['channels'] == []:
        messages.error(
            request,
            'No channels found for experiment: {}'.format(exp))
        return None
    else:
        channels = tuple(resp['channels'])
    return channels


def get_ch_metadata(request, coll, exp, ch):
    add_url = 'collection/{}/experiment/{}/channel/{}'.format(coll, exp, ch)
    url, headers = get_boss_request(request, add_url)
    resp = get_resp_from_boss(request, url, headers)
    if resp == 'authentication failure':
        return resp  # need to handle this better
    return resp


def get_exp_metadata(request, coll, exp):
    add_url = 'collection/{}/experiment/{}'.format(coll, exp)
    url, headers = get_boss_request(request, add_url)
    resp = get_resp_from_boss(request, url, headers)
    if resp == 'authentication failure':
        return resp  # need to handle this better
    return resp


def get_coordinate_frame(request, coll, exp):
    # https://api.theboss.io/v1/coord/:coordinate_frame
    exp_meta = get_exp_metadata(request, coll, exp)
    if exp_meta == 'authentication failure':
        return exp_meta
    coord_frame = exp_meta['coord_frame']
    add_url = 'coord/{}'.format(coord_frame)
    url, headers = get_boss_request(request, add_url)
    resp = get_resp_from_boss(request, url, headers)
    if resp == 'authentication failure':
        return resp  # need to handle this better
    # check that it contains these data otherwise raise exception:
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
    return resp


# helper functions which process data from the Boss or don't interact with
# the Boss:

# this is questionable where this should go - helper file or file that
# gets stuff from the boss
def get_chan_img_data(request, coll, exp, channel, x, y, z):
    base_url, headers = get_boss_request(request)
    headers_blosc = headers
    headers_blosc['Content-Type'] = 'application/blosc-python'

    cut_url = ret_cut_urls(request, base_url, coll, exp, x, y, z, [channel])[0]
    r = get_resp_from_boss(request, cut_url, headers_blosc)
    if r == 'authentication failure' or r == 'incorrect cutout arguments' or r == 'server error':
        return r, cut_url
    data_decomp = blosc.decompress(r.content)

    ch_metadata = get_ch_metadata(request, coll, exp, channel)
    ch_datatype = ch_metadata['datatype']
    data_mat = np.fromstring(data_decomp, dtype=ch_datatype)

    x_rng, y_rng, z_rng = create_voxel_rng(x, y, z)
    # if this is a time series, you need to reshape it differently
    img_data = np.reshape(data_mat,
                          (z_rng[1] - z_rng[0],
                           y_rng[1] - y_rng[0],
                           x_rng[1] - x_rng[0]),
                          order='C')

    return img_data, cut_url


def get_voxel_size(coord_frame):
    x = coord_frame['x_voxel_size']
    y = coord_frame['y_voxel_size']
    z = coord_frame['z_voxel_size']
    return [x, y, z]


def ret_ndviz_channel_part(boss_url, ch_metadata, coll, exp, ch, ch_indx=0):
    if ch_metadata['datatype'] == 'uint16':
        window = 'window=0,10000'
    else:
        window = ''
    if ch_metadata['type'] == 'image':
        chan_type = 'image'
    else:
        chan_type = 'segmentation'

    col_idx = str(ch_indx % 7)  # 7 colors supported by ndviz including white
    opacity = ''
    if ch_indx > 0:
        # 'opacity':0.5
        opacity = '_\'opacity\':0.5'

    visible_option = ''
    if ch_indx > 2:
        visible_option = '_\'visible\':false'

    #{'ch0':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/ailey-dev/Th1eYFP_control_12/ch0?window=0,10000'_'color':2}_'ch1':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/ailey-dev/Th1eYFP_control_12/ch1'_'opacity':0.45_'color':1}}
    ch_link = ''.join(('\'', ch, '\':{\'type\':\'', chan_type, '\'_\'source\':\'boss://',
                       boss_url, coll, '/', exp, '/', ch, '?', window, '\'', opacity, '_\'color\':',
                       col_idx, visible_option, '}'))
    return ch_link


def ret_ndviz_urls(request, base_url, coll, exp,
                   channels, x=None, y=None, z=None):
    # https://viz-dev.boss.neurodata.io/#!%7B%27layers%27:%7B%27synapsinR_7thA%27:%7B%27type%27:%27image%27_%27source%27:%27boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000%27%7D%7D_%27navigation%27:%7B%27pose%27:%7B%27position%27:%7B%27voxelSize%27:[100_100_70]_%27voxelCoordinates%27:[583.1588134765625_5237.650390625_18.5]%7D%7D_%27zoomFactor%27:15.304857247764861%7D%7D
    # unescaped by: http://www.utilities-online.info/urlencode/
    # https://viz-dev.boss.neurodata.io/#!{'layers':{'synapsinR_7thA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/synapsinR_7thA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[583.1588134765625_5237.650390625_18.5]}}_'zoomFactor':15.304857247764861}}
    ndviz_base = 'https://viz.boss.neurodata.io/'
    boss_url = 'https://api.boss.neurodata.io/'
    # error check

    if z is not None:
        z_rng = list(map(int, z.split(':')))
    else:
        z_rng = [0, 1]

    ndviz_urls = []
    z_vals = []
    for z_val in range(z_rng[0], z_rng[1]):
        ch_viz_links = []
        for ch_indx, ch in enumerate(channels):
            ch_metadata = get_ch_metadata(request, coll, exp, ch)
            ch_link = ret_ndviz_channel_part(
                boss_url, ch_metadata, coll, exp, ch, ch_indx)
            ch_viz_links.append(ch_link)

        joined_ndviz_url = ''.join(
            (ndviz_base, '#!{\'layers\':{', '_'.join(ch_viz_links), '}'))

        if x is not None and y is not None:
            x_vals = list(map(int, x.split(':')))
            x_mid = str(round(sum(x_vals) / 2))
            y_vals = list(map(int, y.split(':')))
            y_mid = str(round(sum(y_vals) / 2))
            # add navigation
            joined_ndviz_url = joined_ndviz_url + '_\'navigation\':{\'pose\':{\'position\':{\'voxelCoordinates\':['\
                + x_mid + '_' + y_mid + '_' + str(z_val) + ']}}}}'
        else:
            joined_ndviz_url = joined_ndviz_url + '}'

        ndviz_urls.append(joined_ndviz_url)
        z_vals.append(str(z_val))
    return ndviz_urls, z_vals


def ret_cut_urls(request, base_url, coll, exp, x, y, z, channels):
    res = 0
    cut_urls = []
    for ch in channels:
        JJ = '/'.join(('cutout', coll, exp, ch, str(res), x, y, z))
        ch_metadata = get_ch_metadata(request, coll, exp, ch)
        if ch_metadata['datatype'] == 'uint16':
            window = '?window=0,10000'
        else:
            window = ''
        cut_urls.append(base_url + JJ + '/' + window)
    return cut_urls


def error_check_int_param(vals):
    split_val = vals.split(':')
    try:
        val_chk_list = [str(int(a)) for a in split_val]
        vals_chk = ':'.join(val_chk_list)

        # check here if value is within range of the coord_frame, otherwise,
        # raise an exception

        return vals_chk

    except Exception as e:
        print(e)


def xyz_from_params(q):
    x = error_check_int_param(q.get('x'))
    y = error_check_int_param(q.get('y'))
    z = error_check_int_param(q.get('z'))
    return x, y, z


def process_params(q):
    # validation / data sanitization needed here because it's not being done in the form
    # x_rng_str = q.get('x')

    coll = q.get('coll')
    exp = q.get('exp')
    channels = q.get('channels')
    channels = channels.split(',')

    x, y, z = xyz_from_params(q)

    return coll, exp, x, y, z, channels


def create_voxel_rng(x, y, z):
    x_rng = list(map(int, x.split(':')))
    y_rng = list(map(int, y.split(':')))
    z_rng = list(map(int, z.split(':')))
    return x_rng, y_rng, z_rng


def plot_sgram(request, coll, exp, x, y, z, channels):
    base_url, headers = get_boss_request(request)

    num_ch = len(channels)

    fig = Figure(figsize=(10, 25), dpi=150, facecolor='w', edgecolor='k')
    #fig=plt.figure(figsize=(10, 25), dpi= 150, facecolor='w', edgecolor='k')
    for ch_idx, ch in enumerate(
            channels):  # , exception_handler=exception_handler
        data_mat, _ = get_chan_img_data(request, coll, exp, ch, x, y, z)

        # loop over z and plot them across
        for z_idx in range(data_mat.shape[0]):
            B = data_mat[z_idx, :, :]
            num_z = data_mat.shape[2]
            ax = fig.add_subplot(
                num_ch, num_z + 1, (num_z + 1) * (ch_idx) + (z_idx + 1))
            #plt.subplot(num_ch,num_z+1,(num_z+1)*(ch_idx) + (z_idx+1))
            ax.imshow(B, cmap='gray')
            # ax.xticks([])
            # ax.yticks([])
            # if ch_idx is 0:
            #ax.title('z='+ str(z_idx + z_rng[0]))
            # if z_idx is 0:
            # ax.ylabel(channels[ch_idx])
            if z_idx == data_mat.shape[0] - 1:
                C = np.concatenate(data_mat, axis=1)
                Csum = np.mean(C, axis=1) / 10e3

                ax1 = fig.add_subplot(
                    num_ch, num_z + 1, (num_z + 1) * (ch_idx) + (z_idx + 1) + 1)
                y_idx = np.flip(np.arange(len(Csum)), 0) * .8
                ax1.barh(y_idx, Csum, facecolor='blue')
                #ax1.ylim((min(y_idx), max(y_idx)))
                # plt.xlim(0,1)
                # ax.xticks([])
                # ax.yticks([])
    fig.tight_layout(pad=0, rect=[.02, .02, .98, .98])

    # plt.savefig('synaptogram.png')
    # return render(request, 'synaptogram/sgram.html',context)
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response


def parse_ndviz_url(request, url):
    # example URL:
    #"https://viz-dev.boss.neurodata.io/#!{'layers':{'CR1_2ndA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/CR1_2ndA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[657.4783325195312_1069.4876708984375_11]}}_'zoomFactor':69.80685914923684}}"
    split_url = url.split('/')
    if split_url[2] != 'viz-dev.boss.neurodata.io' and split_url[2] != 'viz.boss.neurodata.io':
        return 'incorrect source', None
    coll = split_url[8]
    exp = split_url[9]

    return coll, exp


def ndviz_units_to_boss(coord_frame, ndviz_voxel, xyz_int):
    # z doesn't change only xy
    z = xyz_int[2]
    xy = xyz_int[0:2]
    # doesn't account for time

    boss_vox_size_xy = [
        coord_frame['x_voxel_size'],
        coord_frame['y_voxel_size']]
    ndviz_voxel_xy = map(float, ndviz_voxel[0:2])

    xy_conv = list(map(lambda a, b, n: round(a / n * b),
                       xy, boss_vox_size_xy, ndviz_voxel_xy))
    xy_conv.append(z)

    return xy_conv
