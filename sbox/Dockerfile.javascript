FROM node:16.20
ADD timeout.sh /timeout.sh
ENTRYPOINT ["/bin/bash","-c","while true; do sleep 1; done"]