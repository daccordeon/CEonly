#!/bin/bash
# script adapted from Lilli and Boris's workaround to view a Jupyter Notebook remotely
read -p 'Please enter the IP:PORT from OzStar (e.g. 192.168.44.13:1234): ' IPPORT
PORT=$(echo $IPPORT| cut -d':' -f 2)
# ozstar is an alias for <username>@ozstar.swin.edu.au set in ~/.ssh/config
ssh -N -L $PORT:$IPPORT ozstar
