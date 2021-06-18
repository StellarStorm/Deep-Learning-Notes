+++
author = "Skylar"
title = "Experiences in Upgrading a DGX Station"
date = "2021-06-18"
tags = ["hardware"]
categories = [
    "deep learning",
]
+++

Recently I've begun maintaining an older-model DGX station (V100  GPUs). This
machine has been in our lab for many years and has been instrumental in our
research in
[fully-automated radiotherapy planning](https://rpa.mdanderson.org/). However,
it hasn't been updated in a long time. Still running the original DGXOS 3
(Ubuntu 16.04) and CUDA libraries, it didn't support our TensorFlow 2
Docker or Singularity images.

The
[Nvidia documentation](https://docs.nvidia.com/dgx/dgx-station-user-guide/index.html#upgrading-dgx-os-software-release)
is rather good in listing the steps needed and possible conflicts during the
upgrade process.
However, I ran into a few unexpected snags that are worth documenting. During
the initial `$ apt full-upgrade` to bring the DGXOS 3 up to the latest packages
within that series, the upgrade process got hung on

```bash
update-alternatives: using /user/bin/dockerd-ce to provide /user/bin/dockerd (dockerd) in auto mode
```

After 15 minutes of waiting with no update, I manually killed the `dockerd`
process with `\$ sudo kill -9 \$(pidof dockerd)` . This let the update finish, but
of course the docker package and a few others were left in a broken state.
This had to be fixed with a `$ sudo dpkg –configure –a`.

Keep an eye on Docker! This won't be the last time it has an issue...

Upgrading to DGXOS 4 was pretty straightforward. There were a few file conflicts
(/etc/systemd/system/docker.service.d/docker-override.conf,
/etc/NetworkManager/NetworkManager.conf, and /etc/ssh/ssh_config) but nothing
unexpected. It's a good idea to keep the local ssh_config of course, but I
updated the others. As it turned out, however, the change of

```code
[ifupddown]
managed=false
```

in NetworkManager.conf meant that the ethernet interface wouldn't connect to the
network after I rebooted into DGXOS 4. Changing this back to `managed=true`
and following up with `$ sudo service network-manager restart` fixed the
connection issue in just a few moments though.

Upgrading to DGXOS 5 was even simpler. For the few file conflicts, it made sense
to accept the package maintainer's version (except still for ssh_config).

Following the login, I got a message that language support was incomplete. This
was a straightforward fix -- just follow the graphical prompts and the packages
were installed in just a few moments.

There were also a lot of temporary repo sources left behind -
`$ sudo apt update` warned of repos being duplicated in multiple files.

Inside /etc/apt/sources.list.d

```code
dgxstation-bionic-r418-cuda-10-1-repo.list
dgxstation-bionic-r418-cuda-10-1-repo.list.distUpgrade
dgxstation.list
dgxstation.list.distUpgrade
dgxstation.list.dpkg-old
dgxstation.list.save
docker.list.distUpgrade
docker.list.dpkg-old
docker.list.save
google-chrome.list.distUpgrade
google-chrome.list.save
nv-rel-upgrade-temp.list
nv-redl-upgrade-temp.list-distUpgrade
ubuntu-esm-infra-list.distUpgrade
```

In addition, there were "sources.list.distUpgrade" and "sources.list.save" in
/etc/apt/.

All of these were moved out of /etc/apt/sources.list.d and into an archive directory.
Most of them referred back to Ubuntu 16.04 (Xenial) or 18.04 (Bionic).
However, "nv-rel-upgrade-temp.list.distUpgrade" was identical to
"nv-rel-upgrade-temp.list". Furthermore, the contents of
"nv-rel-upgrade-temp.list" were distributed to "dgx.list" (Ubuntu sources) and
"cuda-compute-repo.list" (NVidiea CUDA sources).

This left the following files in /etc/apt/sources.list.d/

```code
cuda-compute-repo.list
dgx.list
docker.list
google-chrome.list
ubuntu-esm-infra.list
```

and, of course, the file "/etc/apt/sources.list".

Of these, "docker.list" was disabled during the upgrade process, and I left it
this way as Nvidia
[recommends](https://docs.nvidia.com/dgx/dgx-station-user-guide/index.html#updating-exclusive-dgx-station-software)
avoiding installing docker from any location but their own repos.

Despite the system upgrades, Nvidia-Docker and the Docker CE engine were still
the old versions. Nvidia provides upgrade instructions
[here](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html)
to replace `nvidia-docker` with `nvidia-docker2`. However, each time I tried to
install `nvidia-docker2` as per the instructions, I received this lovely
error

```bash
$ sudo apt install nvidia-docker2
Reading package lists... Done
Building dependency tree
Reading state information... Done
The following additional packages will be installed:
  libnvidia-container-tools libnvidia-container1 nvidia-container-runtime nvidia-container-toolkit
The following packages will be REMOVED:
  nvidia-docker
The following NEW packages will be installed:
  libnvidia-container-tools libnvidia-container1 nvidia-container-runtime nvidia-container-toolkit nvidia-docker2
0 upgraded, 5 newly installed, 1 to remove and 4 not upgraded.
Need to get 0 B/1,471 kB of archives.
After this operation, 9,341 kB disk space will be freed.
Do you want to continue? [Y/n] y
(Reading database ... 296245 files and directories currently installed.)
Removing nvidia-docker (1.0.1-4) ...
Stopping GPU container: 3d7877189077
Committing GPU container changes to e686d36e1dc8-3d7877189077
dpkg: error processing package nvidia-docker (--remove):
installed nvidia-docker package pre-removal script subprocess returned error exit status 1
dpkg: too many errors, stopping
Errors were encountered while processing:
nvidia-docker
Processing was halted because there were too many errors.
E: Sub-process /usr/bin/dpkg returned an error code (1)
```

Could this have been due to the failed Docker update back in the intial stages
that broke a package? I'm really not sure. But in any case, it meant we were
still stuck with the older versions of Docker.

After trying a few different ways to get this to work, I finally went with the
not-so-preferred option of removing the package info.

```bash
sudo rm -rf /var/lib/dpkg/info/nvidia-docker.*
```

Finally, the update worked! The old `nvidia-docker` package was still hanging
around, so I have to manually remove it as well

```bash
sudo apt install nvidia-docker2
sudo apt –purge autoremove nvidia-docker
```

But of course, that wasn't quite the end of it. Nvidia Docker had been updated,
but the Docker Engine itself was left over from Ubuntu Bionic (DGXOS 4).
Uninstalling and reinstalling everything finally brought Docker up to speed.

```bash
sudo apt autoremove nvidia-docker2  # also removed docker-ce
sudo apt install nvidia-docker2 nv-docker-options  # re-installed docker-ce and dependencies
```

After installing Singularity, our DGX Station is now finally able to run the
latest TensorFlow versions!

```bash
20/20 [==============================] - ETA: 0s - loss: 14.8231 - dice_coef_pos: 0.0442 - dice_coef: 0.0300
Epoch 00001: val_loss improved from inf to 14.57323, saving model to /weights/20210618_151214_temp.h5
```
