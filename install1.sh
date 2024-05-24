#!/usr/bin/env python
#!/bin/bash
FEDSCALE_HOME=$(pwd)

echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc
echo alias fedscale=\'bash ${FEDSCALE_HOME}/fedscale.sh\' >> ~/.bashrc