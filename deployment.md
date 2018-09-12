# Install and deployment

## Install

1. Checkout code from github: `git clone https://github.com/neurodata/ndwebtools.git`
1. Initialize submodules: `git submodule update --init --recursive`
1. Create virtual environment & activate
1. Install dependencies: `pip install -r requirements.txt`
1. Create `local_settings.py` file inside mysite/ with the following values:
    ```ini
    SECRET KEY
    DEBUG (True/False)
    ALLOWED_HOSTS (`['*']`)
    auth_uri
    cliend_id
    public_uri
    ```
    
    Note that SECRET KEY can be generated in Python:
    
    ```python
    from django.core.management.utils import get_random_secret_key
    get_random_secret_key()
    ```
    
1. Database initialization
    1. `python manage.py makemigrations bossoidc`
    1. `python manage.py migrate`
1. Test if site works with this command: `python manage.py runserver 0:8080`

## Update

These are for updating production deployment, after the steps below have been followed:

1. From within the ndwebtools repo: `git pull`
    - This should update the git submodule.  If for some reason it doesn't, run: `git submodule update --recursive --remote`
1. Restart uwsgi: `sudo systemctl restart uwsgi`

## Deployment

Deployment (for production) using [uwsgi](https://uwsgi-docs.readthedocs.io/en/latest/) and [nginx](https://www.nginx.com/)

The following instructions are modified from the [tutorial](https://uwsgi-docs.readthedocs.io/en/latest/tutorials/Django_and_nginx.html) for setting up uWSGI with Django (and nginx)

- Installation (systemwide install)

    - `sudo apt install uwsgi`
    - `sudo apt install nginx`
    - drop the uwsgi_params file in your local path (might not be needed): https://github.com/nginx/nginx/blob/master/conf/uwsgi_params

### Nginx config

- create `mysite_nginx.conf` file
    ```apacheconf
    # mysite_nginx.conf

    # the upstream component nginx needs to connect to
    upstream django {
        server unix:///home/ubuntu/uwsgi-ndwebtools/ndwebtools/mysite.sock; # for a file socket
        # server 127.0.0.1:8080; # for a web port socket
    }

    # configuration of the server
    server {
        # the port your site will be served on
        listen      80;
        # the domain name it will serve for
        server_name ndwebtools.neurodata.io 127.0.0.1; # substitute your machine's IP address or FQDN
        charset     utf-8;

        # max upload size
        client_max_body_size 75M;   # adjust to taste

        # Django media
        location /media  {
            alias /home/ubuntu/uwsgi-ndwebtools/ndwebtools/media;  # your Django project's media files - amend as required
        }

        location /static {
            alias /home/ubuntu/uwsgi-ndwebtools/ndwebtools/static; # your Django project's static files - amend as required
        }

        # Finally, send all non-media requests to the Django server.
        location / {
            uwsgi_pass  django;
            include     /home/ubuntu/uwsgi-ndwebtools/ndwebtools/uwsgi_params; # the uwsgi_params file you installed
        }
    }

    server {
        listen       80;
        server_name  ben-dev.neurodata.io ndwt.neurodata.io;

        return 301 $scheme://ndwebtools.neurodata.io$request_uri;

    }

    uwsgi_buffer_size   32k;
    uwsgi_buffers   16 16k;
    ```
- make symbolic link for systemd and nginx
    - `sudo ln -s /home/ubuntu/uwsgi-ndwebtools/ndwebtools/mysite_nginx.conf /etc/nginx/sites-enabled/`

### uWSGI config
- create uwsgi.ini file
    ```ini
    # mysite_uwsgi.ini file
    [uwsgi]

    uid = webuser
    gid = www-data
    base = /home/ubuntu/uwsgi-ndwebtools

    # Django-related settings
    # the base directory (full path)
    chdir           = %(base)/ndwebtools
    # Django's wsgi file
    module          = mysite.wsgi
    # the virtualenv (full path)
    home            = %(base)

    #block size for header
    buffer-size     = 65535

    # process-related settings
    # master
    master          = true
    # maximum number of worker processes
    processes       = 4
    # the socket (use the full path to be safe
    socket          = %(base)/ndwebtools/mysite.sock
    # ... with appropriate permissions - may be needed
    chmod-socket    = 666
    # clear environment on exit
    vacuum          = true
    ```
- make symbolic link for systemd and uwsgi
    - `sudo ln -s ~/uwsgi-ndwebtools/ndwebtools/mysite_uwsgi.ini /etc/uwsgi/apps-enabled/`

### Collect the static files into one location
- `python manage.py collectstatic`

### Start the services
- `sudo systemctl daemon-reload`
- `sudo systemctl start uwsgi`
- `sudo systemctl start nginx`

### Site should be responsive at this point - make it autostart on system reboot
- `sudo systemctl enable uwsgi`
- `sudo systemctl enable nginx`

### Logs
- uwsgi
    - `/var/log/uwsgi/app/mysite_uwsgi.log`
- nginx
    - `/var/log/nginx/access.log`
    - `/var/log/nginx/error.log`