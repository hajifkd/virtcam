# virtcam

A on-demand virtual camera with a virtual background for linux

## Detail

This is yet another program to use [bodypix](https://github.com/tensorflow/tfjs-models/tree/body-pix-v2.0.4/body-pix) to detect a body.
When launched, it observe the virtual camera device using inotify and apply the masking process on demand.

## Install

For modules
```sh
$ sudo apt install v4l2loopback-dkms
$ cat /etc/modprobe.d/v4l2loopback.conf 
options v4l2loopback exclusive_caps=1 video_nr=2
$ cat /usr/lib/modules-load.d/v4l2loopback.conf 
v4l2loopback
```

```sh
$ git clone https://github.com/hajifkd/virtcam.git
$ cd virtcam
$ pip3 install .
$ mkdir -p ~/.config/virtcam
$ cp config.json ~/.config/virtcam
$ cp samplebg.jpg ~/.config/virtcam
$ mkdir -p ~/.config/systemd/user/
$ cp virtcam.service ~/.config/systemd/user/
$ systemctl --user enable virtcam.service
```

