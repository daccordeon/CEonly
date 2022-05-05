#!/bin/bash
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Script (adapted from Lilli and Boris) to join and view/edit a remote Jupyter Notebook
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 
# License:
#     BSD 3-Clause License
# 
#     Copyright (c) 2022, James Gardner.
#     All rights reserved except for those for the gwbench code which remain reserved
#     by S. Borhanian; the gwbench code is included in this repository for convenience.
# 
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
# 
#     1. Redistributions of source code must retain the above copyright notice, this
#        list of conditions and the following disclaimer.
# 
#     2. Redistributions in binary form must reproduce the above copyright notice,
#        this list of conditions and the following disclaimer in the documentation
#        and/or other materials provided with the distribution.
# 
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
# 
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#     DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#     OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#     OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This script is the complement of the command which starts a remote Jupyter notebook, which can be placed in a ~/.alias file:
# jn_remote_start () {
#     # adapted from Lilli's instructions
#     echo '(1) Run on OzStar:'
#     echo 'jn_remote_start'
#     echo 'Note the IP:port (e.g. 192.168.44.1:31234) and 127.0.0.1 '
#     echo '(2) Run on your local machine:'
#     echo 'jn_remote_join'
#     echo 'Enter the IP:port from OzStar when prompted'
#     echo '(3) Open the 127.0.0.1 link from OzStar in your browser'
#     ipnport=$(shuf -i8000-9999 -n1)
#     ipnip=$(hostname -i)
#     ~/.local/bin/jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip
# }

read -p 'Please enter the IP:PORT from OzStar (e.g. 192.168.44.13:1234): ' IPPORT
PORT=$(echo $IPPORT| cut -d':' -f 2)
# ozstar is an alias for <username>@ozstar.swin.edu.au set in ~/.ssh/config
ssh -N -L $PORT:$IPPORT ozstar
