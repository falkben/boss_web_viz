from django import forms
from django.core import validators


class CutoutForm(forms.Form):
    def __init__(self, *args, **kwargs):
        choices = kwargs.pop('channels')
        limits = kwargs.pop('limits')
        res_vals = kwargs.pop('res_vals')

        super(CutoutForm, self).__init__(*args, **kwargs)
        self.fields['channels'] = forms.MultipleChoiceField(
            label='Channels:', choices=[(c, c) for c in choices],
            widget=forms.CheckboxSelectMultiple,
            initial=[c for c in choices])
        # self.fields['channels'] = forms.ChoiceField(label='Channels',
        #     choices=[(c, c) for c in CHOICES])

        self.fields['x_min'] = forms.IntegerField(
            label='x_min', min_value=limits['x_start'], max_value=limits['x_stop'], initial=limits['x_start'])
        self.fields['x_max'] = forms.IntegerField(
            label='x_max', min_value=limits['x_start'], max_value=limits['x_stop'], initial=limits['x_stop'])

        self.fields['y_min'] = forms.IntegerField(
            label='y_min', min_value=limits['y_start'], max_value=limits['y_stop'], initial=limits['y_start'])
        self.fields['y_max'] = forms.IntegerField(
            label='y_max', min_value=limits['y_start'], max_value=limits['y_stop'], initial=limits['y_stop'])

        self.fields['z_min'] = forms.IntegerField(
            label='z_min', min_value=limits['z_start'], max_value=limits['z_stop'], initial=0)
        self.fields['z_max'] = forms.IntegerField(
            label='z_max', min_value=limits['z_start'], max_value=limits['z_stop'], initial=limits['z_stop'])

        self.fields['res_select'] = forms.ChoiceField(
            label='resolution:', choices=[(c, c) for c in res_vals], widget=forms.Select())

    next_action = (
        ('ndviz', 'NeuroData Viz links per slice'),
        ('tiff_stack', 'Download TIFF stack per channel'),
        ('sgram', 'Synaptogram'),
    )
    endpoint = forms.ChoiceField(label='Return:', choices=next_action,
                                 widget=forms.RadioSelect())


# add validations -
#   x_min < x_max
