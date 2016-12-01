#!/usr/bin/env bash

youtube-dl \
  -f 18 \
  -o '%(id)s.%(ext)s' \
  'https://www.youtube.com/watch?v=dM06AMFLsrc'
