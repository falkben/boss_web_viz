import os
import re
import time
import zipfile
from io import BytesIO

import blosc
import matplotlib
import numpy as np
import requests
import tifffile as tiff
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views import generic
from django.contrib.auth import user_logged_in
from django.dispatch import receiver

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from .boss_remote import BossRemote
from .forms import CutoutForm

import ext.neuroglancer.python.neuroglancer as neuroglancer


def load_cmap(filename):
    cmap = []
    with open(filename, 'r') as f:
        cmap = f.read().strip().split(',')
    return cmap


# this is a maximally distinct colormap, sourced from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
CMAP = load_cmap('synaptogram/colormap.csv')


# All the actual views:


def index(request):
    request.session['next'] = 'synaptogram:coll_list'
    return render(request, 'synaptogram/index.html')


@receiver(user_logged_in)
def start_login_events(sender, user, request, **kwargs):
    setup_boss_remote(request)
    set_sess_exp(request)


def setup_boss_remote(request):
    # this is called when the user logs in and creates a boss remote class for the given user
    if 'boss_remote' not in request.session:
        request.session['boss_remote'] = BossRemote(request)


def set_sess_exp(request):
    # we set the session expiration to match the bearer token expiration
    id_token = request.session.get('id_token')
    if id_token is not None:  # if admin interfact we aren't using KeyCloak
        epoch_time_KC = id_token['exp']
        epoch_time_loc = round(time.time())  # + time.timezone
        new_exp_time = epoch_time_KC - epoch_time_loc
        request.session.set_expiry(new_exp_time)


@login_required
def coll_list(request):
    collections = request.session['boss_remote'].get_collections()

    context = {'collections': collections}
    return render(request, 'synaptogram/coll_list.html', context)


@login_required
def exp_list(request, coll):
    experiments = request.session['boss_remote'].get_experiments(coll)
    if experiments is None:
        messages.error(
            request, 'No experiments found for collection: {}'.format(coll))
        return redirect(reverse('synaptogram:coll_list'))

    context = {'coll': coll, 'experiments': experiments}
    return render(request, 'synaptogram/exp_list.html', context)


@login_required
def cutout(request, coll, exp):
    boss_remote = request.session['boss_remote']
    # we need the channels to fill the form
    channels = boss_remote.get_channels(coll, exp)
    if channels is None:
        messages.error(
            request, 'No channels found for experiment: {}'.format(exp))
        return redirect(reverse('synaptogram:exp_list', args={coll}))

    exp_info = boss_remote.get_exp_info(coll, exp)
    res_vals = list(range(exp_info['num_hierarchy_levels']))

    # getting the coordinate frame limits for the experiment:
    coord_frame = boss_remote.get_coordinate_frame(coll, exp, exp_info)
    # important stuff out of coord_frame:
    # "x_start": 0,
    # "x_stop": 1000,
    # "y_start": 0,
    # "y_stop": 1000,
    # "z_start": 0,
    # "z_stop": 500

    # actual metadata in the BOSS for the experiment
    exp_meta_keyvals = boss_remote.get_exp_metadata(coll, exp)

    # merge the coord_frame data and some experiment metadata together
    exp_data = coord_frame
    copy_keys = ['creator', 'description', 'hierarchy_method',
                 'num_hierarchy_levels', 'num_time_samples',
                 'time_step', 'time_step_unit']
    exp_data.update({key: exp_info[key] for key in copy_keys})
    exp_data.update(exp_meta_keyvals)

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:

        form = CutoutForm(request.POST, channels=channels,
                          limits=coord_frame, res_vals=res_vals)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            x = str(form.cleaned_data['x_min']) + \
                ':' + str(form.cleaned_data['x_max'])
            y = str(form.cleaned_data['y_min']) + \
                ':' + str(form.cleaned_data['y_max'])
            z = str(form.cleaned_data['z_min']) + \
                ':' + str(form.cleaned_data['z_max'])

            res = form.cleaned_data['res_select']

            channels = form.cleaned_data['channels']

            pass_params_d = {'coll': coll, 'exp': exp, 'x': x,
                             'y': y, 'z': z, 'channels': ','.join(channels), 'res': res}
            pass_params = '&'.join(['%s=%s' % (key, value)
                                    for (key, value) in pass_params_d.items()])
            params = '?' + pass_params

            # redirect to a new URL:
            end_path = form.cleaned_data['endpoint']
            if end_path == 'sgram':
                return HttpResponseRedirect(
                    reverse('synaptogram:sgram') + params)
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
                              limits=coord_frame, res_vals=res_vals)
        else:
            form = CutoutForm(channels=channels,
                              limits=coord_frame, res_vals=res_vals)

    context = {'form': form, 'coll': coll, 'exp': exp, 'channels': channels,
               'exp_data': sorted(exp_data.items())}
    return render(request, 'synaptogram/cutout.html', context)


@login_required
def get_ndviz_url(request, coll, exp, channel):
    # this generates an ndviz url for the channel/experiment/collection as inputs
    boss_remote = request.session['boss_remote']

    if channel is None:
        channels = boss_remote.get_channels(coll, exp)
    else:
        channels = [channel]
    url, _ = ret_ndviz_urls(request, coll, exp, channels)
    return redirect(url[0])


@login_required
def ndviz_url_list(request):
    boss_remote = request.session['boss_remote']
    q = request.GET
    coll, exp, x, y, z, channels = process_params(q)

    urls, z_vals = ret_ndviz_urls(request, coll, exp, channels, x, y, z)

    x_int = list(map(int, x.split(':')))

    channel_ndviz_list = zip(z_vals, urls)

    # voxel sizes are used to set the zoom factor for each URL
    coord_frame = boss_remote.get_coordinate_frame(coll, exp)
    voxel_sizes = get_voxel_size(coord_frame)

    context = {
        'channel_ndviz_list': channel_ndviz_list,
        'coll': coll,
        'exp': exp,
        'x': x_int,
        'voxel_sizes': voxel_sizes}
    return render(request, 'synaptogram/ndviz_url_list.html', context)


@login_required
def tiff_stack(request):
    q = request.GET
    coll, exp, x, y, z, channels = process_params(q)
    res = q.get('res')

    urls = []
    for ch in channels:
        # create links to go to a method that will download the TIFF images
        # inside each channel
        urls.append(reverse('synaptogram:tiff_stack_channel',
                            args=(coll, exp, x, y, z, ch, res)))

    # or package the images and create links for the images
    channels_arg = ','.join(channels)

    return render(request, 'synaptogram/tiff_url_list.html', {'urls': urls,
                                                              'coll': coll, 'exp': exp, 'x': x, 'y': y, 'z': z, 'channels': channels_arg, 'res': res})


@login_required
def tiff_stack_channel(request, coll, exp, x, y, z, channel, res):
    img_data, cut_url = get_chan_img_data(
        request, coll, exp, channel, x, y, z, res)
    obj = BytesIO()
    tiff.imsave(obj, img_data, metadata={'cut_url': cut_url})

    fname = '{}_{}_{}_{}_{}_{}_{}.tiff'.format(
        coll, exp, x, y, z, channel, res).replace(':', '_')
    response = HttpResponse(obj.getvalue(), content_type='image/TIFF')
    response['Content-Disposition'] = 'attachment; filename="' + fname + '"'
    return response


@login_required
def zip_tiff_stacks(request, coll, exp, x, y, z, channels, res):
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
                request, coll, exp, ch, x, y, z, res)
            if img_data == 'authentication failure' or img_data == 'incorrect cutout arguments':
                raise Exception
            fn = 'media/{}_{}_{}_{}_{}_{}_{}.tiff'.format(
                coll, exp, x, y, z, ch, res).replace(':', '_')
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

    x = ':'.join(request.GET.get('xextent').split(','))
    y = ':'.join(request.GET.get('yextent').split(','))
    coords = request.GET.get('coords').split(',')
    z_int = int(float(coords[-1]))
    z = '{:d}:{:d}'.format(z_int, z_int + 1)

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
    boss_remote = request.session['boss_remote']
    ch_info = boss_remote.get_ch_info(coll, exp, channel)
    ch_perms = boss_remote.get_permissions(coll, exp, channel)
    downsample_status = boss_remote.get_downsample_status(coll, exp, channel)
    ch_info['downsample_status'] = downsample_status['status']

    keys = ['min_I', 'max_I']
    for k in keys:
        val = boss_remote.get_ch_metadata_key(coll, exp, channel, k)
        if val:
            ch_info[k] = val
        else:
            break

    perm_sets = ch_perms['permission-sets']

    ndviz_url, _ = ret_ndviz_urls(request, coll, exp, [channel])

    return render(request, 'synaptogram/channel_detail.html',
                  {'coll': coll, 'exp': exp, 'channel': channel, 'channel_props': ch_info, 'permissions': perm_sets,
                   'ndviz_url': ndviz_url[0]})


@login_required
def start_downsample(request, coll, exp, channel):
    boss_remote = request.session['boss_remote']
    boss_remote.start_downsample(coll, exp, channel)
    return HttpResponseRedirect(
        reverse('synaptogram:channel_detail', args=(coll, exp, channel)))


@login_required
def stop_downsample(request, coll, exp, channel):
    boss_remote = request.session['boss_remote']
    boss_remote.stop_downsample(coll, exp, channel)
    return HttpResponseRedirect(
        reverse('synaptogram:channel_detail', args=(coll, exp, channel)))


# helper functions which process data from the Boss or don't interact with
# the Boss:

def adjust_downsample(res, xy):
    if res > 0:
        xy_adjust = []
        for ext in xy:
            ext_int = [int(aa) for aa in ext.split(':')]
            xy_adjust.append(
                ':'.join([str(round(xx / (2**res))) for xx in ext_int]))
        return xy_adjust
    else:
        return xy

# this is questionable where this should go - helper file or file that
# gets stuff from the boss


def ret_cut_urls(coll, exp, x, y, z, channels, res):
    cut_urls = []
    for ch in channels:
        JJ = '/'.join(('cutout', coll, exp, ch, str(res), x, y, z))
        cut_urls.append(JJ)
    return cut_urls


def get_chan_img_data(request, coll, exp, channel, x, y, z, res):
    boss_remote = request.session['boss_remote']

    res = int(res)
    x, y = adjust_downsample(res, [x, y])

    cut_url = ret_cut_urls(coll, exp, x, y, z, [channel], res)[0]
    r = boss_remote.get(cut_url, {'Accept': 'application/blosc'})

    data_decomp = blosc.decompress(r.content)

    ch_info = boss_remote.get_ch_info(coll, exp, channel)
    ch_datatype = ch_info['datatype']
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


def ret_ndviz_layer(boss_url, ch_metadata, coll, exp, ch):
    if ch_metadata['datatype'] == 'uint16':
        if 'min_I' in ch_metadata:  # we have max/min values as BOSS metadata
            window = 'window={},{}'.format(
                ch_metadata['min_I'], ch_metadata['max_I'])
        else:
            window = 'window=0,10000'
    else:
        window = ''

    # construct source url
    source_url = 'boss://{}/{}/{}/{}?{}'.format(
        boss_url, coll, exp, ch, window)

    if ch_metadata['type'] == 'image':
        layer = neuroglancer.ImageLayer(
            source=source_url,
            blend="additive",
            opacity=1,
        )
    else:
        layer = neuroglancer.SegmentationLayer(
            source=source_url,
            selectedAlpha=.4)

    return layer


def ret_ndviz_urls(request, coll, exp, channels, x=None, y=None, z=None):
    boss_url = 'https://api.boss.neurodata.io'

    # we query the boss for info on the channel
    boss_remote = request.session['boss_remote']

    if z is not None:
        z_rng = list(map(int, z.split(':')))
    else:
        z_rng = [0, 1]

    ndviz_urls = []
    z_vals = []

    layers = []
    ch_infos = []
    img_chs = 0
    for ch_indx, ch in enumerate(channels):
        # for each channel we get some metadata
        ch_info = boss_remote.get_ch_info(coll, exp, ch)

        # we look for previously specified window values
        # if annotation data, we never window
        if ch_info['datatype'] != 'uint64':
            img_chs += 1  # used for setting the colormap
            keys = ['min_I', 'max_I']
            for k in keys:
                val = boss_remote.get_ch_metadata_key(coll, exp, ch, k)
                if val:
                    ch_info[k] = val
                else:
                    break

        ch_layer = ret_ndviz_layer(boss_url, ch_info, coll, exp, ch)
        layers.append(ch_layer)
        ch_infos.append(ch_info)

    set_nav = False
    if x is not None and y is not None:
        x_vals = list(map(int, x.split(':')))
        x_mid = str(round(sum(x_vals) / 2))
        y_vals = list(map(int, y.split(':')))
        y_mid = str(round(sum(y_vals) / 2))
        set_nav = True

    # sort the segmentation layers after the image layers (good for opacity)
    segs = list(map(lambda ch_i: ch_i['datatype'] == 'uint64', ch_infos))
    idx = sorted(range(len(segs)), key=segs.__getitem__)

    for z_val in range(z_rng[0], z_rng[1]):
        state = neuroglancer.ViewerState()
        state.layout = 'xy'

        # add in each layer
        visible = True
        skip_chs = 0
        for i, (layer, ch, ch_info) in enumerate(zip([layers[i] for i in idx], [channels[i] for i in idx], [ch_infos[i] for i in idx])):
            # disable visibility for channel index > 2 in list after sorting
            if i > 2:
                visible = False

            kwargs = {}
            if ch_info['datatype'] != 'uint64':
                kwargs['color'] = CMAP[i % len(CMAP) - skip_chs]
            else:
                skip_chs += 1

            state.layers.append(
                name=ch,
                layer=layer,
                visible=visible,
                **kwargs,
            )
        if set_nav:
            state.voxel_coordinates = [x_mid, y_mid, z_val]

        ndviz_urls.append(neuroglancer.to_url(state))
        z_vals.append(str(z_val))
    return ndviz_urls, z_vals


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
    num_ch = len(channels)

    fig = Figure(figsize=(10, 25), dpi=150, facecolor='w', edgecolor='k')
    #fig=plt.figure(figsize=(10, 25), dpi= 150, facecolor='w', edgecolor='k')
    for ch_idx, ch in enumerate(
            channels):  # , exception_handler=exception_handler
        data_mat, _ = get_chan_img_data(request, coll, exp, ch, x, y, z, 0)

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
    # "https://viz-dev.boss.neurodata.io/#!{'layers':{'CR1_2ndA':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/kristina15/image/CR1_2ndA?window=0,10000'}}_'navigation':{'pose':{'position':{'voxelSize':[100_100_70]_'voxelCoordinates':[657.4783325195312_1069.4876708984375_11]}}_'zoomFactor':69.80685914923684}}"
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
