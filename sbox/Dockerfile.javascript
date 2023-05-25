FROM node:16.20
ADD timeout.sh /timeout.sh
ENTRYPOINT ["/timeout.sh"]